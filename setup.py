from setuptools import setup, find_packages

setup(
    name="genai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.2",
        "azure-search-documents>=11.4.0",
    ],
    python_requires=">=3.8",
)
