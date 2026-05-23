import os
import json
import string
import requests
import datasets
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def download_file(url, filename):
    """Download a file in chunks to avoid timeout issues with large files."""
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download.")
        return
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Done downloading {filename}.")
    except requests.exceptions.Timeout:
        print(f"Timeout while downloading {filename}. Try again or download manually.")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        raise


def download_data():
    download_file(
        "https://people.cs.kuleuven.be/~thomas.bauwens/irse_documents_2026_recipes.parquet",
        "irse_documents_2026_recipes.parquet"
    )
    download_file(
        "https://people.cs.kuleuven.be/~thomas.bauwens/irse_queries_2026_recipes.json",
        "irse_queries_2026_recipes.json"
    )

def load_dataset():
    return datasets.load_dataset("parquet", data_files="./irse_documents_2026_recipes.parquet")['train']

def load_queries():
    return json.load(open("./irse_queries_2026_recipes.json", "r"))

def download_nltk_resources():
    """Download required NLTK resources if not already present."""
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for resource in resources:
        nltk.download(resource, quiet=True)  # quiet=True suppresses the output

def build_and_save_testcases_for_vocabulary(dataset):
    doc_lists = {
            "name": [],
            "name_ingredients": [],
            "name_tags": [],
            "name_ingredients_tags": [],
            "all": []
        }
    mwe_list = {
            "name": [],
            "name_ingredients": [],
            "name_tags": [],
            "name_ingredients_tags": [],
            "all": []
        }
    for example in dataset:
        name = example["name"]
        ingredients = example["ingredients"]
        tags = example["tags"]
        description = example["description"]
        steps = example["steps"]

        doc_lists["name"].append(name)
        doc_lists["name_ingredients"].append(name + " " + ingredients)
        doc_lists["name_tags"].append(name + " " + tags)
        doc_lists["name_ingredients_tags"].append(name + " " + ingredients + " " + tags)
        doc_lists["all"].append(name + " " + ingredients + " " + tags + " " + description + " " + steps)

        mwe_list["name"].append(name)
        mwe_list["name_ingredients"].append(name + " " + ingredients)
        mwe_list["name_tags"].append(name + " " + tags)
        mwe_list["name_ingredients_tags"].append(name + " " + ingredients + " " + tags)
        mwe_list["all"].append(name + " " + ingredients + " " + tags + " " + description + " " + steps)
    return doc_lists, mwe_list

def preprocess(list_of_text):
  preprocessed_text = []
  for text in list_of_text:
    text = text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    text = text.lower() #convert to lower case
    words = word_tokenize(text) #tokenize the text
    words = [word for word in words if word not in stopwords.words('english')] #remove stopwords
    preprocessed_text.append(words)
  return preprocessed_text

def make_mwe_analyzer(mwes):
    """Custom analyzer that merges MWE bigrams before TF-IDF."""
    stop = set(stopwords.words('english'))
    def analyzer(text):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        words = [w for w in word_tokenize(text) if w not in stop]
        merged = []
        i = 0
        while i < len(words):
            if i < len(words) - 1 and (words[i], words[i+1]) in mwes:
                merged.append(words[i] + '_' + words[i+1])
                i += 2
            else:
                merged.append(words[i])
                i += 1
        return merged
    return analyzer

def find_mwes2(preprocessed_docs, min_freq=20, pmi_min=3.0, pmi_max=8.0, top_n=150):
    all_words = [word for doc in preprocessed_docs for word in doc]
    finder = BigramCollocationFinder.from_words(all_words)
    finder.apply_freq_filter(min_freq)

    scored = finder.score_ngrams(BigramAssocMeasures.pmi)

    # Filter by PMI range
    mwes = {bigram for bigram, score in scored if pmi_min <= score <= pmi_max}

    # Take top_n by frequency instead of PMI to avoid proper noun bias
    freq_filtered = [(bigram, finder.ngram_fd[bigram]) for bigram in mwes]
    freq_filtered.sort(key=lambda x: x[1], reverse=True)

    return set(bigram for bigram, _ in freq_filtered[:top_n])

def preprocess_fast(list_of_text):
    stop = set(stopwords.words('english'))
    result = []
    for text in list_of_text:
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        words = [w for w in text.split() if w not in stop]
        result.append(words)
    return result

def clean_queries_fun(queries):
    clean_queries = [query["q"] for query in queries["queries"]]
    return clean_queries
