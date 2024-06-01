import os
import re
import fitz
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import TokenTextSplitter


load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# neo4j_uri = os.getenv("NEO4J_URI")
# neo4j_username = os.getenv("NEO4J_USERNAME")
# neo4j_password = os.getenv("NEO4J_PASSWORD")
#
# os.environ["OPENAI_API_KEY"] = openai_api_key
# os.environ["NEO4J_URI"] = neo4j_uri
# os.environ["NEO4J_USERNAME"] = neo4j_username
# os.environ["NEO4J_PASSWORD"] = neo4j_password

graph = Neo4jGraph(url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"],
                           password=os.environ["NEO4J_PASSWORD"])
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm_transformer = LLMGraphTransformer(llm=llm)

# List of articles to process
articles = [
    "7 July 2005 London bombings",
    "Beslan school siege",
    "Moscow theater hostage crisis",
    "Qahtaniyah bombings",
    "Camp Speicher massacre",
    "2011 Norway attacks",
    "March 2015 Sanaa mosque bombings",
    "May 2017 Kabul bombing",
    "2017 London Bridge attack",
    "14 October 2017 Mogadishu bombings"
]

# Define chunking strategy
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
#
# for article in articles:
#     print(f"Processing article: {article}")
#
#     # Read the Wikipedia article
#     raw_documents = WikipediaLoader(query=article).load()
#
#     # Split the documents into chunks
#     documents = text_splitter.split_documents(raw_documents[:3])
#     graph_documents = llm_transformer.convert_to_graph_documents(documents)
#     graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
#
#     # Process the documents
#     for i, doc in enumerate(documents):
#         print(f"Document {i + 1}:")
#         print(doc)
#         print("---")
#
#     print("---")
article = "Beslan school siege"
print(f"Processing article: {article}")

    # Read the Wikipedia article
raw_documents = WikipediaLoader(query=article).load()

    # Split the documents into chunks
documents = text_splitter.split_documents(raw_documents[:3])
graph_documents = llm_transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

    # Process the documents
for i, doc in enumerate(documents):
    print(f"Document {i + 1}:")
    print(doc)
    print("---")

# print("---")