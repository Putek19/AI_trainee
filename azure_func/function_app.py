import azure.functions as func
import logging
import json
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.RAG.ai_search_langchain import RAGSystem

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

        # Get response from RAG system
        result = rag_system.ask_question(query)

        # Log the result for debugging
        logging.info(f"RAG result: {json.dumps(result, indent=2)}")

        # Ensure sources is always a list
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


@app.route(route="rag-interface", methods=["GET"])
def serve_interface(req: func.HttpRequest) -> func.HttpResponse:
    try:
        html_path = Path(__file__).parent / "static" / "index.html"
        with open(html_path, "r", encoding="utf-8") as file:
            content = file.read()
        return func.HttpResponse(body=content, mimetype="text/html", status_code=200)
    except Exception as e:
        logging.error(f"Error serving interface: {str(e)}")
        return func.HttpResponse("Error serving interface", status_code=500)
