# Document Q&A System Frontend

This is a Streamlit-based frontend for the Document Q&A system that interacts with Azure Functions backend.

## Features

- Upload new PDF documents
- Ask questions about the uploaded documents
- View chat history
- Clean and intuitive user interface

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install from root requirements.txt)
- Running Azure Functions backend

## Running the Application

1. Make sure your Azure Functions backend is running locally (default: http://localhost:7071)
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. The application will open in your default web browser (default: http://localhost:8501)

## Usage

1. **Upload Documents:**
   - Use the sidebar to upload PDF documents
   - Click the "Upload Document" button to process the document

2. **Ask Questions:**
   - Type your question in the text input field
   - Click "Ask" to get an answer
   - View the chat history below

## Note

Make sure your Azure Functions backend is running before using the frontend. The application expects the following endpoints to be available:
- `http://localhost:7071/api/upload_document` - For document uploads
- `http://localhost:7071/api/chat` - For Q&A functionality
