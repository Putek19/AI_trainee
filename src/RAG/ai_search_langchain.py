import os
import logging
import json
from typing import List, Optional
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector"
os.environ["AZURESEARCH_FIELDS_CONTENT"] = "content"
os.environ["AZURESEARCH_FIELDS_TAG"] = "meta_json_string"

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
        Load documents from a file and upload them to Azure Search.
        """
        logger.info(f"Loading document: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Only .txt and .pdf files are supported")

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)

        # Format documents for Azure Search
        formatted_docs = []
        for i, doc in enumerate(split_docs):
            formatted_doc = Document(
                page_content=doc.page_content,
                metadata={
                    "url": "default",
                    "filepath": file_path,
                    "title": os.path.basename(file_path),
                    "meta_json_string": json.dumps(
                        {"chunk": i + 1, "page": doc.metadata.get("page", "n/a")}
                    ),
                },
            )
            formatted_docs.append(formatted_doc)

        try:
            self.vector_store.add_documents(formatted_docs)
            logger.info(f"Uploaded {len(formatted_docs)} documents to Azure Search")
        except Exception as e:
            logger.error(f"Failed to upload documents to Azure Search: {e}")
            raise

        return split_docs

    def load_documents_from_memory(self, file_obj) -> List[Document]:
        """
        Load documents from a file-like object in memory.

        Args:
            file_obj: A file-like object (e.g., BytesIO) containing the document data

        Returns:
            List[Document]: List of processed document chunks
        """
        logger.info(f"Loading document from memory: {file_obj.name}")

        ext = os.path.splitext(file_obj.name)[1].lower()

        if ext == ".txt":
            text = file_obj.read().decode("utf-8")
            documents = [
                Document(page_content=text, metadata={"source": file_obj.name})
            ]
        elif ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            import tempfile

            # PyPDFLoader requires a file path, so we need to save temporarily
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(file_obj.read())
                temp_file.flush()
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()
                os.unlink(temp_file.name)  # Clean up temp file
        elif ext == ".csv":
            import pandas as pd

            df = pd.read_csv(file_obj)
            documents = [
                Document(
                    page_content=row.to_string(),
                    metadata={"source": file_obj.name, "row": i},
                )
                for i, row in df.iterrows()
            ]
        else:
            raise ValueError("Only .txt, .pdf, and .csv files are supported")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)

        # Format documents for Azure Search
        formatted_docs = []
        for i, doc in enumerate(split_docs):
            formatted_doc = Document(
                page_content=doc.page_content,
                metadata={
                    "url": "default",
                    "filepath": file_obj.name,
                    "title": os.path.basename(file_obj.name),
                    "meta_json_string": json.dumps(
                        {"chunk": i + 1, "page": doc.metadata.get("page", "n/a")}
                    ),
                },
            )
            formatted_docs.append(formatted_doc)

        try:
            self.vector_store.add_documents(formatted_docs)
            logger.info(f"Uploaded {len(formatted_docs)} documents to Azure Search")
        except Exception as e:
            logger.error(f"Failed to upload documents to Azure Search: {e}")
            raise

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

        # Check if there are any documents in the vector store
        try:
            qa_chain = self._create_qa_chain()
            result = qa_chain.invoke({"query": query})
        except Exception as e:
            logger.error(f"Error during question processing: {str(e)}")
            return {
                "answer": "I apologize, but I couldn't process your question. Please make sure documents are loaded into the system first.",
                "sources": [],
            }

        answer = result["result"]
        sources = result.get("source_documents", [])

        # Process sources with better fallback values and error handling
        source_list = []
        for i, doc in enumerate(sources, 1):
            try:
                metadata = doc.metadata

                # Get source name with fallbacks
                source_candidates = [
                    metadata.get("source"),
                    metadata.get("title"),
                    metadata.get("filepath"),
                    metadata.get("file_path"),
                    metadata.get("document_name"),
                    f"Document {i}",
                ]
                source_name = next(
                    (s for s in source_candidates if s), "Unknown Source"
                )

                # Get page number with fallback
                page = metadata.get("page", metadata.get("chunk", i))

                # Try to parse meta_json_string if it exists
                try:
                    if "meta_json_string" in metadata:
                        meta_data = json.loads(metadata["meta_json_string"])
                        if isinstance(meta_data, dict):
                            if "page" in meta_data:
                                page = meta_data["page"]
                            if "chunk" in meta_data:
                                page = meta_data["chunk"]
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse meta_json_string for source {i}")

                source_list.append({"source": str(source_name), "page": str(page)})
                logger.debug(f"Processed source {i}: {source_name} (page: {page})")

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
            print("2. Load documents")
            print("3. Exit")

            choice = input("Choose an option (1-3): ").strip()

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
                logger.info("Loading files...")
                file_path = input("Enter the file path (.txt or .pdf): ").strip()
                if file_path and os.path.isfile(file_path):
                    try:
                        self.load_documents_from_file(file_path)
                        print("📄 Documents loaded successfully.")
                    except Exception as e:
                        logger.error(f"Error loading documents: {str(e)}")
                        print(f"⚠️ Error loading documents: {str(e)}")
                else:
                    logger.warning("Invalid file path provided")
                    print(
                        "⚠️ Invalid file path. Please provide a valid .txt or .pdf file."
                    )
            elif choice == "3":
                logger.info("Exiting interactive mode")
                print("👋 Program terminated.")
                break
            else:
                logger.warning(f"Invalid option selected: {choice}")
                print("⚠️ Invalid option. Choose 1, 2, or 3.")


if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.interactive_mode()
