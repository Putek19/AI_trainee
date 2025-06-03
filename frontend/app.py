import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="Document Q&A System", page_icon="📚", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def make_api_request(
    endpoint, method="POST", json_data=None, files=None, data=None, headers=None
):
    """Helper function to make API requests with better error handling"""
    try:
        if method == "POST":
            if data:  # For file upload
                response = requests.post(
                    f"http://localhost:7071/api/{endpoint}", data=data, headers=headers
                )
            else:  # For JSON requests
                response = requests.post(
                    f"http://localhost:7071/api/{endpoint}", json=json_data
                )

            # Debug information
            st.sidebar.write("Debug Info:")
            st.sidebar.write(f"Status Code: {response.status_code}")
            try:
                st.sidebar.write(f"Response: {response.json()}")
            except:
                st.sidebar.write(f"Raw Response: {response.text}")

            return response
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def main():
    st.title("📚 Document Q&A System")

    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload New Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "csv"],
            help="Upload PDF, TXT, or CSV files to be processed by the RAG system",
        )

        if uploaded_file is not None:
            if st.button("Upload Document"):
                with st.spinner("Uploading and processing document..."):
                    # Get file content and name
                    file_content = uploaded_file.getvalue()
                    filename = uploaded_file.name

                    # Set up headers for binary content
                    headers = {
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(file_content)),
                    }

                    # Make the request
                    response = make_api_request(
                        f"upload_file?filename={requests.utils.quote(filename)}",  # URL encode the filename
                        method="POST",
                        data=file_content,
                        headers=headers,
                    )

                    if response and response.status_code == 200:
                        result = response.json()
                        st.success(
                            f"Document uploaded and processed successfully! {result.get('message', '')}"
                        )
                    else:
                        st.error(
                            "Error uploading document. Please check the debug information in the sidebar."
                        )

    # Main chat interface
    st.header("Ask Questions About Your Documents")

    # Question input
    user_question = st.text_input("Enter your question:")

    if st.button("Ask"):
        if user_question:
            with st.spinner("Getting answer..."):
                # Call Azure Function for Q&A
                response = make_api_request(
                    "ask_rag", json_data={"query": user_question}
                )

                if response and response.status_code == 200:
                    try:
                        answer_data = response.json()

                        # Display the answer
                        st.subheader("Answer:")
                        st.write(answer_data.get("answer", "No answer received"))

                        # Display sources if available
                        sources = answer_data.get("sources", [])
                        if sources:
                            st.subheader("Sources:")
                            for source in sources:
                                st.write(
                                    f"- {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')})"
                                )

                        # Add to chat history
                        st.session_state.chat_history.append(
                            {
                                "question": user_question,
                                "answer": answer_data.get(
                                    "answer", "No answer received"
                                ),
                                "sources": sources,
                            }
                        )

                    except json.JSONDecodeError as e:
                        st.error(f"Error parsing response: {str(e)}")
                        st.write("Raw response:", response.text)
                else:
                    st.error(
                        "Error getting response from the server. Check debug information in sidebar."
                    )

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for chat in reversed(st.session_state.chat_history):
            with st.expander(f"Q: {chat['question'][:100]}..."):
                st.write("**Question:**")
                st.write(chat["question"])
                st.write("**Answer:**")
                st.write(chat["answer"])
                if chat.get("sources"):
                    st.write("**Sources:**")
                    for source in chat["sources"]:
                        st.write(
                            f"- {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')})"
                        )
                st.markdown("---")


if __name__ == "__main__":
    main()
