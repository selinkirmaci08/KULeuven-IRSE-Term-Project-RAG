import json
import datasets
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import time
import subprocess
R_NUMBER_SEED = 1080271
DOCS_TO_ADD   = 2000
MODEL_NAME    = "sentence-transformers/multi-qa-mpnet-base-cos-v1"


def download_data():
    files = [
        "acl_anthology_queries.json",
        "acl_anthology_full.parquet",
        "acl_anthology_queries.parquet"
    ]

    base_url = "https://lagom.cs.kuleuven.be/cs/courses/irse/"
    max_retries = 5
    wait_seconds = 10

    for filename in files:
        if os.path.exists(filename):
            print(f"{filename} already exists, skipping.")
            continue

        for attempt in range(1, max_retries + 1):
            print(f"Downloading {filename} (attempt {attempt}/{max_retries})...")
            result = subprocess.run(["wget", "-q", base_url + filename], capture_output=True)

            if result.returncode == 0 and os.path.exists(filename):
                print(f"✓ {filename} downloaded successfully.")
                break
            else:
                print(f"✗ Failed. Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
        else:
            print(f"✗ Could not download {filename} after {max_retries} attempts.")

def load_data():
    query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
    all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]

    random_documents = all_documents.shuffle(seed=R_NUMBER_SEED).take(DOCS_TO_ADD)
    anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)
    anthology_sample.to_parquet("./anthology_sample.parquet")

    with open("./acl_anthology_queries.json", "r") as f:
        queries = json.load(f)

    return anthology_sample, queries


def build_index_mapping(anthology_sample):
    """Build idx <-> acl_id mappings."""
    idx_to_acl_id = [ex["acl_id"] for ex in anthology_sample]
    acl_id_to_idx = {acl_id: i for i, acl_id in enumerate(idx_to_acl_id)}
    return idx_to_acl_id, acl_id_to_idx


def get_document_text(example, mode="title_abstract"):
    title     = example.get("title", "")     or ""
    abstract  = example.get("abstract", "")  or ""
    full_text = example.get("full_text", "") or ""
    author    = example.get("author", "")    or ""

    if mode == "title":
        return title
    elif mode == "title_abstract":
        return title + " [SEP] " + abstract
    elif mode == "title_abstract_author":
        return title + " [SEP] " + abstract + " " + author
    elif mode == "all":
        return title + " [SEP] " + abstract + " " + full_text[:500]
    return title + " [SEP] " + abstract


def load_sentence_transformer(model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model


def encodeDocuments(model, docs):
    print("Encode documents")
    return model.encode(docs, show_progress_bar=True, normalize_embeddings=True)


def encodeQueries(model, queries):
    print("Encode queries")
    return model.encode(queries, show_progress_bar=True, normalize_embeddings=True)


def chunkText(title, text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start + chunk_size]
        # prepend title to every chunk for context
        chunks.append(title + " [SEP] " + " ".join(chunk))
        start += chunk_size - overlap
    return chunks


def build_chunk_index(anthology_sample):
    all_chunks = []
    chunk_to_acl_id = []

    for example in anthology_sample:
        if not example["full_text"]:
            continue
        title = example.get("title", "") or ""
        for chunk in chunkText(title, example["full_text"]):
            all_chunks.append(chunk)
            chunk_to_acl_id.append(example["acl_id"])

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks, chunk_to_acl_id


def encode_chunks(model, all_chunks, batch_size=256):
    return model.encode(all_chunks, show_progress_bar=True, batch_size=batch_size, normalize_embeddings=True)


def nearest_neighbors(similarities, top_k=5):
    import torch
    results, result_scores = [], []
    for row in similarities:
        top_k_idx = torch.argsort(row, descending=True)[:top_k]
        results.append(top_k_idx.tolist())
        result_scores.append(row[top_k_idx].tolist())
    return results, result_scores


def map_chunks_to_docs(chunk_results, chunk_to_acl_id, top_k=10):
    doc_results = []
    for query_chunks in chunk_results:
        seen = []
        for chunk_idx in query_chunks:
            acl_id = chunk_to_acl_id[chunk_idx]
            if acl_id not in seen:
                seen.append(acl_id)
            if len(seen) == top_k:
                break
        doc_results.append(seen)
    return doc_results


def map_chunks_to_docs_with_best_chunk(chunk_results, chunk_scores, all_chunks, chunk_to_acl_id, top_k=10):
    doc_results = []
    doc_best_chunks = []
    doc_st_scores = []  # store original ST scores

    for query_chunks, query_scores in zip(chunk_results, chunk_scores):
        seen = {}
        seen_scores = {}

        for chunk_idx, score in zip(query_chunks, query_scores):
            acl_id = chunk_to_acl_id[chunk_idx]
            if acl_id not in seen:
                seen[acl_id] = all_chunks[chunk_idx]
                seen_scores[acl_id] = score  # store actual score
            if len(seen) == top_k:
                break

        doc_results.append(list(seen.keys()))
        doc_best_chunks.append(seen)
        doc_st_scores.append(seen_scores)  # {acl_id -> st score}

    return doc_results, doc_best_chunks, doc_st_scores


def td_idf(docs, queries):
    vectorizer = TfidfVectorizer(
        norm=None,
        stop_words='english',
        min_df=2, max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    tfidf_matrix_queries = vectorizer.transform(queries)
    return tfidf_matrix, tfidf_matrix_queries


def nearest_neighbors_for_tdidf(docs, queries, top_k=10):
    all_scores = cosine_similarity(queries, docs)
    results = []

    for score_row in all_scores:
        top_k_idx = np.argsort(score_row)[::-1][:top_k]
        results.append(top_k_idx)

    return results