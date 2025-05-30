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


def make_api_request(endpoint, method="POST", json_data=None, files=None):
    """Helper function to make API requests with better error handling"""
    try:
        if method == "POST":
            response = requests.post(
                f"http://localhost:7071/api/{endpoint}", json=json_data, files=files
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
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Upload Document"):
                # Save the file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Upload to Azure Function
                with open("temp.pdf", "rb") as f:
                    response = make_api_request("upload_document", files={"file": f})

                if response and response.status_code == 200:
                    st.success("Document uploaded successfully!")
                else:
                    st.error(
                        f"Error uploading document. Please check the debug information in the sidebar."
                    )

                # Clean up temp file
                os.remove("temp.pdf")

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
                        answer = response.json()
                        # Add to chat history
                        st.session_state.chat_history.append(
                            {
                                "question": user_question,
                                "answer": answer.get("answer", "No answer received"),
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
    st.subheader("Chat History")
    for chat in reversed(st.session_state.chat_history):
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.markdown("---")


if __name__ == "__main__":
    main()
