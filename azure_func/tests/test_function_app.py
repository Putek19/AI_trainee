import pytest
import json
from azure.functions import HttpRequest
from function_app import ask_rag


def test_ask_rag_valid_query():
    # Arrange
    test_query = "What is Azure Search?"
    test_body = json.dumps({"query": test_query}).encode("utf-8")

    # Create a mock HTTP request
    req = HttpRequest(
        method="POST",
        url="/api/ask_rag",
        body=test_body,
        params={},
        headers={"Content-Type": "application/json"},
    )

    # Act
    response = ask_rag(req)

    # Assert
    assert response.status_code == 200
    response_body = json.loads(response.get_body())
    assert "status" in response_body
    assert response_body["status"] == "success"
    assert "query" in response_body
    assert response_body["query"] == test_query
    assert "answer" in response_body
    assert "sources" in response_body


def test_ask_rag_missing_query():
    # Arrange
    test_body = json.dumps({}).encode("utf-8")

    # Create a mock HTTP request
    req = HttpRequest(
        method="POST",
        url="/api/ask_rag",
        body=test_body,
        params={},
        headers={"Content-Type": "application/json"},
    )

    # Act
    response = ask_rag(req)

    # Assert
    assert response.status_code == 400
    assert (
        "Please provide a 'query' in the request body." in response.get_body().decode()
    )
