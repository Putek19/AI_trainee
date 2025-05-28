import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector"
from langchain_community.vectorstores.azuresearch import AzureSearch


# Configure logging with more specific settings
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Set logging levels for other modules to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)


class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with Azure configurations."""
        load_dotenv()

        # Azure OpenAI Configuration
        self.azure_openai_deployment = "gpt-4o"
        self.azure_openai_api_key = os.getenv("API_OPEN_AI_KEY")
        self.azure_openai_api_version = "2024-12-01-preview"
        self.azure_openai_endpoint = os.getenv("OPEN_AI_ENDPOINT")
        self.azure_embedding_deployment = os.getenv("EMBEDDING_MODEL_NAME")

        # Azure Search Configuration
        self.azure_search_endpoint = os.getenv("SEARCH_AI_ENDPOINT")
        self.azure_search_api_key = os.getenv("SEARCH_AI_KEY")
        self.azure_search_index = os.getenv("SEARCH_AI_INDEX_NAME")

        self._initialize_components()
        logger.info("RAG System initialized successfully")

    def _initialize_components(self):
        """Initialize LangChain components."""
        self.embeddings = AzureOpenAIEmbeddings(
            openai_api_version=self.azure_openai_api_version,
            azure_endpoint=self.azure_openai_endpoint,
            azure_deployment=self.azure_embedding_deployment,
            api_key=self.azure_openai_api_key,
        )

        self.llm = AzureChatOpenAI(
            deployment_name=self.azure_openai_deployment,
            openai_api_key=self.azure_openai_api_key,
            azure_endpoint=self.azure_openai_endpoint,
            openai_api_version=self.azure_openai_api_version,
            temperature=0,
        )

        self.vector_store = AzureSearch(
            azure_search_endpoint=self.azure_search_endpoint,
            azure_search_key=self.azure_search_api_key,
            index_name=self.azure_search_index,
            embedding_function=self.embeddings.embed_query,
        )

        self.retriever = self.vector_store.as_retriever(k=5)
        logger.debug("Components initialized successfully")

    def load_documents_from_file(self, file_path: str) -> List[Document]:
        """
        Load and split documents from a file into chunks for RAG processing.

        Args:
            file_path (str): Path to the document file (.txt or .pdf)

        Returns:
            List[Document]: List of processed document chunks
        """
        logger.info(f"Loading document: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            error_msg = "Only .txt and .pdf files are supported"
            logger.error(error_msg)
            raise ValueError(error_msg)

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        clean_docs = [
            Document(page_content=doc.page_content, metadata={}) for doc in split_docs
        ]

        self.vector_store.add_documents(split_docs)
        logger.info(f"Loaded and split {len(split_docs)} chunks")
        return split_docs

    def _build_prompt_template(self) -> PromptTemplate:
        """Build a prompt template for the RAG system."""
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

    def _create_qa_chain(self) -> RetrievalQA:
        """Create a RetrievalQA chain for question answering."""
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self._build_prompt_template()},
            return_source_documents=True,
        )

    def ask_question(self, query: str) -> dict:
        """
        Ask a question using the RAG system.

        Args:
            query (str): The question to ask

        Returns:
            dict: Contains 'answer' and 'sources' keys with the response and source documents
        """
        logger.info(f"Processing question: {query}")
        qa_chain = self._create_qa_chain()
        result = qa_chain.invoke({"query": query})

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Process sources with better fallback values
        source_list = []
        for i, doc in enumerate(sources, 1):
            try:
                # Get source name with fallbacks
                source_candidates = [
                    doc.metadata.get("source"),
                    doc.metadata.get("title"),
                    doc.metadata.get("filepath"),
                    doc.metadata.get("file_path"),
                    doc.metadata.get("document_name"),
                    f"Document {i}",
                ]
                source_name = next(
                    (s for s in source_candidates if s), "Unknown Source"
                )

                # Get page number with fallback
                page = doc.metadata.get("page")
                if page is None:
                    page = doc.metadata.get("chunk", i)

                source_list.append({"source": str(source_name), "page": str(page)})

                # Log source details for debugging
                logger.debug(f"Processed source {i}: {source_name} (page: {page})")
                logger.debug(f"Full metadata: {doc.metadata}")

            except Exception as e:
                logger.warning(f"Error processing source {i}: {str(e)}")
                source_list.append({"source": f"Document {i}", "page": str(i)})

        logger.info(f"Processed {len(source_list)} sources successfully")
        return {"answer": answer, "sources": source_list}

    def interactive_mode(self):
        """Run the RAG system in interactive mode."""
        logger.info("Starting interactive mode")
        while True:
            print("\n🔍 LangChain RAG System")
            print("1. Ask a question")
            print("2. Exit")

            choice = input("Choose an option (1-2): ").strip()

            if choice == "1":
                query = input("Enter your question: ").strip()
                if query:
                    result = self.ask_question(query)
                    print("\n📌 Answer:")
                    print(result["answer"])
                    print("\n📚 Sources:")
                    for source in result["sources"]:
                        print(f"{source['source']} - Page {source['page']}")
                else:
                    logger.warning("Empty question provided")
                    print("⚠️ Question cannot be empty.")

            elif choice == "2":
                logger.info("Exiting interactive mode")
                print("👋 Program terminated.")
                break

            else:
                logger.warning(f"Invalid option selected: {choice}")
                print("⚠️ Invalid option. Choose 1 or 2.")


if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.interactive_mode()
