from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from part1.data import *


def preprocessDocMemoryEfficient(docs,mwes):
    vectorizer = TfidfVectorizer(
    norm=None,
    analyzer=make_mwe_analyzer(mwes),
    stop_words='english',
    min_df=2, max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    return tfidf_matrix, vectorizer

def preprocessQueriesMemoryEfficient(allQueries, vectorizer):
    return vectorizer.transform(allQueries)

def nearest_neighbors(docs, queries, top_k=5, metric="cosine"):
    if metric == "cosine":
        all_scores = cosine_similarity(queries, docs)
        ascending = False
    elif metric == "dot":
        all_scores = (queries @ docs.T).toarray()
        ascending = False
    elif metric == "manhattan":
        all_scores = manhattan_distances(queries, docs)
        ascending = True
    elif metric == "euclidean":
        all_scores = euclidean_distances(queries, docs)
        ascending = True

    results = []
    scores = []

    for score_row in all_scores:
        if ascending:
            top_k_idx = np.argsort(score_row)[:top_k]
        else:
            top_k_idx = np.argsort(score_row)[::-1][:top_k]
        results.append(top_k_idx)
        scores.append(score_row[top_k_idx])

    return results, scores

def nearest_neighbors_threshold(docs, queries, threshold=0.1):
    all_scores = cosine_similarity(queries, docs)

    results = []
    scores = []

    for score_row in all_scores:
        top_k_idx = np.where(score_row >= threshold)[0]
        top_k_idx = top_k_idx[np.argsort(score_row[top_k_idx])[::-1]]
        results.append(top_k_idx)
        scores.append(score_row[top_k_idx])

    return results, scores

def compute_metrics(queries, all_ranked, top_k=None):
    precisions, recalls, f1s, average_precisions = [], [], [], []
    total_tp, total_retrieved, total_relevant = 0, 0, 0

    for i, query in enumerate(queries["queries"]):
        gold_ids = set(r[0] for r in query["r"])

        if not gold_ids:
            continue

        ranked_slice = all_ranked[i] if top_k is None else all_ranked[i][:top_k]
        ranked_ids = [int(doc_idx) + 1 for doc_idx in ranked_slice]
        retrieved = set(ranked_ids)

        tp = len(retrieved & gold_ids)
        precision = tp / len(retrieved) if retrieved else 0
        recall = tp / len(gold_ids) if gold_ids else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        hits, ap = 0, 0.0
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in gold_ids:
                hits += 1
                ap += hits / rank
        average_precisions.append(ap / len(gold_ids))

        total_tp += tp
        total_retrieved += len(retrieved)
        total_relevant += len(gold_ids)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    micro_precision = total_tp / total_retrieved if total_retrieved else 0
    micro_recall = total_tp / total_relevant if total_relevant else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0

    MAP = np.mean(average_precisions)

    print(f"── Macro averages ──")
    print(f"  Precision : {macro_precision:.4f}")
    print(f"  Recall    : {macro_recall:.4f}")
    print(f"  F1        : {macro_f1:.4f}")
    print(f"── Micro averages ──")
    print(f"  Precision : {micro_precision:.4f}")
    print(f"  Recall    : {micro_recall:.4f}")
    print(f"  F1        : {micro_f1:.4f}")
    print(f"── MAP ─────────────")
    print(f"  MAP       : {MAP:.4f}")

    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1, MAP

def tryDifferentVocabulary(doc_lists, clean_queries, queries,mwes):
    for mode, docs in doc_lists.items():
        print(f"\n-------Mode: {mode} -------")
        tfidf_matrix_docs, vectorizer = preprocessDocMemoryEfficient(docs,mwes)
        tfidf_matrix_queries = preprocessQueriesMemoryEfficient(clean_queries, vectorizer)
        result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, top_k=10)
        compute_metrics(queries, result, top_k=10)

def calculate():
    print("Load Data")
    download_nltk_resources()
    download_data()
    dataset = load_dataset()
    queries = load_queries()
    docs, mwe_list = build_and_save_testcases_for_vocabulary(dataset)
    mwe_docs = preprocess_fast(mwe_list['name'])
    mwes = find_mwes2(mwe_docs)

    print(list(mwes)[:10])
    print(len(mwes))


    print("Process Data")
    clean_queries = clean_queries_fun(queries)
    documents = docs['name']
    tfidf_matrix_docs, vectorizer = preprocessDocMemoryEfficient(documents,mwes)
    tfidf_matrix_queries = preprocessQueriesMemoryEfficient(clean_queries, vectorizer)


    print("Test different distance metrics")
    print("------cosine similarity------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries)
    compute_metrics(queries, result, top_k=5)

    print("------dot distance------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, metric="dot")
    compute_metrics(queries, result, top_k=5)

    print("------manhattan distance------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, metric="manhattan")
    compute_metrics(queries, result, top_k=5)

    print("------euclidean distance------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, metric="euclidean")
    compute_metrics(queries, result, top_k=5)

    print("Test different top-k values")
    print("------k=5------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, top_k=5)
    compute_metrics(queries, result, top_k=5)
    print("------k=10------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, top_k=10)
    compute_metrics(queries, result, top_k=10)
    print("------k=15------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, top_k=15)
    compute_metrics(queries, result, top_k=15)
    print("------k=20------")
    result, scores = nearest_neighbors(tfidf_matrix_docs, tfidf_matrix_queries, top_k=20)
    compute_metrics(queries, result, top_k=20)

    print("------threshold based retrieval (threshold=0.2)------")
    result, scores = nearest_neighbors_threshold(tfidf_matrix_docs, tfidf_matrix_queries, threshold=0.2)
    avg_retrieved = np.mean([len(r) for r in result])
    print(f"Average documents retrieved per query: {avg_retrieved:.2f}")
    compute_metrics(queries, result)

    print("------threshold based retrieval (threshold=0.3)------")
    result, scores = nearest_neighbors_threshold(tfidf_matrix_docs, tfidf_matrix_queries, threshold=0.3)
    avg_retrieved = np.mean([len(r) for r in result])
    print(f"Average documents retrieved per query: {avg_retrieved:.2f}")
    compute_metrics(queries, result)

    print("Test different vocabularies")
    tryDifferentVocabulary(docs, clean_queries, queries, mwes)
    print("END")

if __name__ == "__main__":
    calculate()