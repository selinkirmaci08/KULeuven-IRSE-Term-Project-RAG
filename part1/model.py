import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from part1.calculations import *
import os


def getPromptForQuery(queryIndex, result, queries, dataset):
    retrieved_docs = []

    for doc_id in result[queryIndex]:
        doc = dataset[int(doc_id)]
        text = f"""
  Name: {doc['name']}
  Description: {doc['description']}
  Ingredients: {doc['ingredients']}
  Steps: {doc['steps']}
  """
        retrieved_docs.append(text)
    formatted_docs = "\n\n".join(
        [f"Recipe {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"""
      <s>[INST]
      You are a helpful cooking assistant.

      Use ONLY the recipes and the information provided below to answer the question.
      If the answer is not in the recipes, say "Not found in recipes."

      Question:
      {queries[queryIndex]}

      Recipes:
      {formatted_docs}

      Return a clear and concise answer.
      [/INST]
      """
    return prompt

def getPromptVague(queryIndex, results, retrieval_queries, dataset):
    retrieved_docs = []

    for i, doc_idx in enumerate(results[queryIndex]):
        doc = dataset[int(doc_idx)]
        retrieved_docs.append(f"Recipe {i+1}: {doc['name']}. {doc['description']}")

    formatted_docs = "\n".join(retrieved_docs)

    prompt = f"""<s>[INST]
            You are a cooking assistant. Here are some recipes:
            
            {formatted_docs}
            
            Answer this: {retrieval_queries[queryIndex]}
            [/INST]
            """
    return prompt

def getPromptForQueryWithScores(queryIndex, results, scores, retrieval_queries, dataset):
    retrieved_docs = []

    for doc_idx, score in zip(results[queryIndex], scores[queryIndex]):
        doc = dataset[int(doc_idx)]
        retrieved_docs.append(f"""
Name: {doc['name']}
Relevance Score: {score:.4f}
Description: {doc['description']}
Ingredients: {doc['ingredients']}
Steps: {doc['steps']}
""")

    formatted_docs = "\n\n".join(
        [f"Recipe {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    prompt = f"""
<s>[INST]
You are a helpful cooking assistant.

Use ONLY the recipes and the information provided below to answer the question.
The recipes are ranked by relevance score (higher = more relevant).
If the answer is not in the recipes, say "Not found in recipes."

Question:
{retrieval_queries[queryIndex]}

Recipes:
{formatted_docs}

Return a clear and concise answer.
[/INST]
"""
    return prompt


def callModel(model, tokenizer, prompt, do_sample=True):
    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded_prompt = encoded_prompt.to("cuda")
    generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=do_sample)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])


def getPrompotListWithScroes(dataset, result, retrieval_queries, scores):
    promptList = []
    for i in range(len(result)):
        prompt = getPromptForQueryWithScores(i, result, scores, retrieval_queries, dataset)
        promptList.append(prompt)
    return promptList


def getPromptListForRetrievalQueries(dataset, result, retrieval_queries):
    promptList = []
    for i in range(len(result)):
        prompt = getPromptForQuery(i, result, retrieval_queries, dataset)
        promptList.append(prompt)
    return promptList


def printRetrievedDocs(queryIndex, result, queries, dataset):
    print(f"\n-----------------------Query {queryIndex}-----------------------------")
    print(f"Query: {queries[queryIndex]}\n")

    for i, doc_id in enumerate(result[queryIndex]):
        doc = dataset[int(doc_id)]
        print(f"Recipe {i+1}:")
        print(f"  Name: {doc['name']}")
        print(f"  Official ID: {doc['official_id']}")
        print(f"  Description: {doc['description']}")
        print(f"  Tags: {doc['tags']}")
        print(f"  Ingredients: {doc['ingredients']}")
        print(f"  Steps: {doc['steps']}")
        print("-" * 40)


def printAllRetrievedDocs(dataset, result, retrieval_queries):
    for i in range(len(result)):
        printRetrievedDocs(i, result, retrieval_queries, dataset)


def getIrrelaventContextPrompt() -> str:
    irrelevant_context = """
    Richard Gary Brautigan (January 30, 1935 – c. September 16, 1984)
    was an American novelist, poet, and short story writer. A prolific writer,
    he wrote throughout his life and published ten novels, two collections of
    short stories, and four books of poetry. Brautigan's work has been published
    both in the United States and internationally throughout Europe, Japan,
    and China. He is best known for his novels Trout Fishing in America (1967),
    In Watermelon Sugar (1968), and The Abortion: An Historical Romance 1966 (1971).
    """

    prompt_irrelevant = f"""
    <s>[INST]
    You are a helpful cooking assistant.
    Use ONLY the recipes below to answer the question.
    If the answer is not in the recipes, say "Not found in recipes."

    Question: lunch baked at 300 degrees?
    Recipes: {irrelevant_context}

    Return a clear and concise answer.
    [/INST]
    """
    return prompt_irrelevant


def runModel():
    # load model
    os.environ["HF_TOKEN"] = "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"
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto'
    )

    retrieval_queries_drawbacks = [
        "afternoon recipe with eggs",
        "choclate cake recpie",
        "authentic thai street food",
        "I need a recipe with less than 5 ingredients",
        "chicken recipe with no sauce at all"
    ]

    retrieval_queries_good = [
        "vanilla cake",
        "vegetarian lasagna but easy"
    ]

    reasoning_queries = [
        "Which of these recipes takes the least amount of time to prepare?",
        "If I am allergic to dairy, which of these recipes can I still make?",
        "Which recipe requires the most ingredients?",
        "Which recipes could be made vegan with minimal substitutions?",
        "Which recipe is both gluten-free and can be made in under 30 minutes?"
    ]

    # load data
    download_nltk_resources()
    download_data()
    dataset = load_dataset()
    docs, mwe_list = build_and_save_testcases_for_vocabulary(dataset)
    mwe_docs = preprocess_fast(mwe_list['name'])
    mwes = find_mwes2(mwe_docs)

    # process data
    documents = docs['name']
    tfidf_matrix_docs, vectorizer = preprocessDocMemoryEfficient(documents, mwes)
    tf_idf_value_retrieval_queries = preprocessQueriesMemoryEfficient(retrieval_queries_drawbacks, vectorizer)
    result, scores = nearest_neighbors(tfidf_matrix_docs, tf_idf_value_retrieval_queries, top_k=3)

    tf_idf_value_reasoning_queries = preprocessQueriesMemoryEfficient(reasoning_queries, vectorizer)
    resultForReasoning, scoresForReasoning = nearest_neighbors(tfidf_matrix_docs, tf_idf_value_reasoning_queries, top_k=3)

    tf_idf_value_retrieval_queries_good = preprocessQueriesMemoryEfficient(retrieval_queries_good, vectorizer)
    goodResults, goodScores = nearest_neighbors(tfidf_matrix_docs, tf_idf_value_retrieval_queries_good, top_k=3)

    promptListWithoutScores = getPromptListForRetrievalQueries(dataset, goodResults, retrieval_queries_good)
    promptListWithScores = getPrompotListWithScroes(dataset, goodResults, retrieval_queries_good, goodScores)
    promptListForReasoning = getPromptListForRetrievalQueries(dataset, resultForReasoning, reasoning_queries)

    printAllRetrievedDocs(dataset, result, retrieval_queries_drawbacks)

    for i in range(2):
        print(f"""-----------------------Query {i}-----------------------------""")
        callModel(model, tokenizer, promptListWithoutScores[i])

    for i in range(2):
        print(f"""-----------------------Query {i} for prompts with scores-----------------------------""")
        callModel(model, tokenizer, promptListWithScores[i])

    for i in range(5):
        print(f"""-----------------------Query {i} for reasining-----------------------------""")
        callModel(model, tokenizer, promptListForReasoning[i])

    forNoRelevantRecipe = getPromptListForRetrievalQueries(dataset, result, retrieval_queries_drawbacks)
    callModel(model, tokenizer, forNoRelevantRecipe[4])

    prompt_irrelevant = getIrrelaventContextPrompt()
    callModel(model, tokenizer, prompt_irrelevant)

