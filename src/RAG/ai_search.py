import os
import json

from typing import List, Dict, Any
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
from datetime import datetime


load_dotenv()


MODEL_NAME = "gpt-4o"
DEPLOYMENT = "gpt-4o"
endpoint = os.getenv("SEARCH_AI_ENDPOINT")
index_name = os.getenv("SEARCH_AI_INDEX_NAME")
api_key = os.getenv("SEARCH_AI_KEY")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("OPEN_AI_ENDPOINT"),
    api_key=os.getenv("API_OPEN_AI_KEY"),
)

search_client = SearchClient(
    endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key)
)


def get_embeddings(text: str) -> List[float]:
    """
    Get embeddings for the given text using Azure OpenAI.
    """
    if not isinstance(text, str):
        raise ValueError("Input to get_embeddings must be a string")
    response = client.embeddings.create(model=embedding_model_name, input=text)
    return response.data[0].embedding


def search_documents(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for documents based on the query using either semantic or vector search.
    """
    # if semantic:

    #     results = search_client.search(
    #         search_text=query,
    #         query_type="semantic",
    #         semantic_configuration_name="default",
    #         select=["id", "content", "title", "url", "filepath", "meta_json_string"],
    #         top=top_k
    #     )

    embedding = get_embeddings(query)
    vector = VectorizedQuery(vector=embedding, top=top_k, fields="contentVector")
    results = search_client.search(
        search_text=query,
        vector_queries=[vector],
        select=["id", "content", "title", "url", "filepath", "meta_json_string"],
        top=top_k,
    )
    documents = [
        {
            "id": res.get("id", ""),
            "content": res.get("content", ""),
            "url": res.get("url", ""),
            "filepath": res.get("filepath", ""),
            "title": res.get("title", ""),
            "meta_json_string": res.get("meta_json_string", ""),
            "score": res.get("@search.score", 0),
            "source": res.get("url", "Unknown"),
        }
        for res in results
    ]

    for i, res in enumerate(documents):
        print(f"{i+1}. Document: {res['title']} (score =  {res['score']:.4f})")
    return documents


def build_prompt(documents: List[Dict[str, Any]], query: str) -> str:
    """
    Build a prompt for the AI model using the retrieved documents and the query.
    """
    prompt = f"Query: {query}\n\n"
    prompt += "Retrieved Documents:\n"

    for doc in documents:
        if doc.get("title"):
            prompt += f"Title: {doc['title']}\n"
        prompt += f"Content: {doc['content']}\n"
        prompt += f"URL: {doc['url']}\n"
        prompt += f"Score: {doc['score']:.4f}\n\n"
    prompt += (
        "You are an AI assistant that answers questions strictly based on the provided context documents. "
        "Guidelines:"
        "- Be clear, concise, and accurate in your response."
        "- Support your answers with references to the document numbers when possible (e.g., Document 1)."
        "- If the documents contain conflicting information, acknowledge the discrepancy and explain it if possible."
        "- Only use information available in the context documents — do not speculate or make assumptions."
        "- If the context lacks sufficient information, clearly state that the answer cannot be determined."
    )

    return prompt


def ask_gpt4(prompt: str) -> str:
    """
    Send the prompt to GPT-4 and return the response.
    """
    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def save_query_results_to_notebook(
    query: str,
    documents: List[Dict[str, Any]],
    filepath: str = "notebooks/queries.ipynb",
):
    """
    Save query and its results as a new cell in a Jupyter notebook.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    notebook = nbformat.v4.new_notebook()

    timestamp = datetime.now().isoformat()
    json_content = {"query": query, "timestamp": timestamp, "results": documents}

    code_cell = new_code_cell(
        f"# Zapytanie z dnia {timestamp}\n\nimport json\nquery_result = {json.dumps(json_content, indent=2)}"
    )
    notebook.cells.append(code_cell)

    with open(filepath, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)

    print(f"Wyniki zapytania zapisano do {filepath}")


def answer_question_with_sources(question: str, top_k: int = 5) -> None:
    """
    Main function to answer a question using retrieved documents.

    Args:
        question (str): The question to answer.
        top_k (int): Number of top documents to retrieve.
    """
    print(f"Szukanie dokumentów dla zapytania: {question}")
    documents = search_documents(query=question, top_k=top_k)

    if not documents:
        print("Nie znaleziono dokumentów.")
        return

    prompt = build_prompt(documents, question)
    print("Wysyłanie zapytania do GPT-4...")
    answer = ask_gpt4(prompt)

    print("Odpowiedź GPT-4:\n")
    print(answer)

    print("\nŹródła:")
    for doc in documents:
        print(f"- {doc.get('title') or doc.get('filepath') or doc.get('url')}")
    save_query_results_to_notebook(question, documents)


if __name__ == "__main__":
    print("Witaj w systemie wyszukiwania dokumentów!")
    user_question = input("Zadaj pytanie: ")
    answer_question_with_sources(user_question)
