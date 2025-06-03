import azure.functions as func
import logging
import json
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from RAG.ai_search_langchain import RAGSystem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


rag_system = RAGSystem()


@app.route(route="ask_rag", methods=["POST"])
def ask_rag(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("RAG query function processed a request.")

    try:
        req_body = req.get_json()
        query = req_body.get("query")

        if not query:
            return func.HttpResponse(
                "Please provide a 'query' in the request body.", status_code=400
            )

        result = rag_system.ask_question(query)

        logging.info(f"RAG result: {json.dumps(result, indent=2)}")

        sources = result.get("sources", [])
        if not isinstance(sources, list):
            sources = [sources] if sources else []

        response_data = {
            "status": "success",
            "query": query,
            "answer": result["answer"],
            "sources": sources,
        }

        # Log the final response
        logging.info(f"Sending response: {json.dumps(response_data, indent=2)}")

        return func.HttpResponse(
            json.dumps(response_data), mimetype="application/json", status_code=200
        )

    except ValueError as e:
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": "Invalid JSON in request body",
                    "error": str(e),
                }
            ),
            status_code=400,
            mimetype="application/json",
        )
    except Exception as e:
        logging.error(f"Error processing RAG query: {str(e)}")
        return func.HttpResponse(
            json.dumps(
                {"status": "error", "message": "Internal server error", "error": str(e)}
            ),
            status_code=500,
            mimetype="application/json",
        )


# @app.route(route="rag-interface", methods=["GET"])
# def serve_interface(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         html_path = Path(__file__).parent / "static" / "index.html"
#         with open(html_path, "r", encoding="utf-8") as file:
#             content = file.read()
#         return func.HttpResponse(body=content, mimetype="text/html", status_code=200)
#     except Exception as e:
#         logging.error(f"Error serving interface: {str(e)}")
#         return func.HttpResponse("Error serving interface", status_code=500)


@app.route(route="upload_file", methods=["POST"])
def upload_file(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("File upload function processed a request.")

    try:
        # Get the file from the request
        file_data = req.get_body()
        file_name = req.params.get("filename")

        if not file_data or not file_name:
            return func.HttpResponse(
                json.dumps(
                    {"status": "error", "message": "No file or filename provided"}
                ),
                status_code=400,
                mimetype="application/json",
            )

        temp_dir = Path(__file__).parent / "temp_uploads"
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / file_name
        with open(temp_file_path, "wb") as f:
            f.write(file_data)

        try:

            docs = rag_system.load_documents_from_file(str(temp_file_path))

            temp_file_path.unlink()

            return func.HttpResponse(
                json.dumps(
                    {
                        "status": "success",
                        "message": f"Successfully processed {len(docs)} document chunks from {file_name}",
                        "chunks_processed": len(docs),
                    }
                ),
                mimetype="application/json",
                status_code=200,
            )
        finally:

            if temp_file_path.exists():
                temp_file_path.unlink()

    except Exception as e:
        logging.error(f"Error processing file upload: {str(e)}")
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": "Error processing file upload",
                    "error": str(e),
                }
            ),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="http_trigger", auth_level=func.AuthLevel.ANONYMOUS)
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    name = req.params.get("name")
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get("name")

    if name:
        return func.HttpResponse(
            f"Hello, {name}. This HTTP triggered function executed successfully."
        )
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200,
        )
