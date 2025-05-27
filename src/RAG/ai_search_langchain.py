import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


load_dotenv()


AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_API_KEY = os.getenv("API_OPEN_AI_KEY")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_ENDPOINT = os.getenv("OPEN_AI_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_MODEL_NAME")

AZURE_SEARCH_ENDPOINT = os.getenv("SEARCH_AI_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("SEARCH_AI_KEY")
AZURE_SEARCH_INDEX = os.getenv("SEARCH_AI_INDEX_NAME")

os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector"
from langchain_community.vectorstores.azuresearch import AzureSearch


embeddings = AzureOpenAIEmbeddings(
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_key=AZURE_OPENAI_API_KEY,
)

llm = AzureChatOpenAI(
    deployment_name=AZURE_OPENAI_DEPLOYMENT,
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0,
)

vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    azure_search_key=AZURE_SEARCH_API_KEY,
    index_name=AZURE_SEARCH_INDEX,
    embedding_function=embeddings.embed_query,
)

retriever = vector_store.as_retriever(k=5)


def load_documents_from_file(file_path: str) -> list[Document]:
    """
    Load and split documents from a file into chunks for RAG processing.
    """
    print(f"📂 Ładowanie dokumentu: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("❌ Obsługiwane są tylko pliki .txt i .pdf")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    clean_docs = [
        Document(page_content=doc.page_content, metadata={}) for doc in split_docs
    ]

    vector_store.add_documents(split_docs)
    print(f"✅ Załadowano i podzielono {len(split_docs)} fragmentów.")
    return split_docs


def build_prompt_template() -> PromptTemplate:
    """
    Build a prompt template for the RAG system."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an AI assistant that answers questions strictly based on the provided context documents.

Context:
{context}

Question: {question}

Instructions:
- Be concise and accurate.
- Use document numbers if relevant (e.g., Document 1).
- Don't speculate; say if the answer can't be determined from the context.
""",
    )


def create_qa_chain() -> RetrievalQA:
    """
    Create a RetrievalQA chain for question answering using the retriever and LLM.
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": build_prompt_template()},
        return_source_documents=True,
    )


def ask_question(query: str) -> str:
    """
    Ask a question using the RAG system and return the answer along with source documents."""
    qa_chain = create_qa_chain()
    result = qa_chain.invoke({"query": query})

    answer = result["result"]
    sources = result.get("source_documents", [])

    print("\n📌 Odpowiedź:")
    print(answer)

    print("\n📚 Źródła:")
    for i, doc in enumerate(sources, 1):
        meta = (
            doc.metadata.get("title")
            or doc.metadata.get("source")
            or doc.metadata.get("filepath")
            or "Unknown"
        )
        print(f"{i}. {meta}")

    return answer


def main_menu():
    while True:
        print("\n🔍 LangChain RAG System")
        # print("1. Załaduj dokument")
        print("1. Zadaj pytanie")
        print("2. Wyjście")

        choice = input("Wybierz opcję (1-2): ").strip()

        # if choice == "1":
        #     file_path = input("Podaj ścieżkę do pliku (.txt lub .pdf): ").strip()
        #     try:
        #         load_documents_from_file(file_path)
        #     except Exception as e:
        #         print(f"❌ Błąd przy ładowaniu dokumentów: {e}")

        if choice == "1":
            query = input("Zadaj pytanie: ").strip()
            if query:
                ask_question(query)
            else:
                print("⚠️ Pytanie nie może być puste.")

        elif choice == "2":
            print("👋 Zakończono działanie programu.")
            break

        else:
            print("⚠️ Niepoprawna opcja. Wybierz 1 lub 2.")


if __name__ == "__main__":
    main_menu()
