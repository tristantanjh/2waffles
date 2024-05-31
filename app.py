import panel as pn
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import helper as hp
import io
import re
import fitz
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import os
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.documents import Document
from langchain_core.runnables import ConfigurableField, RunnableParallel, RunnablePassthrough
import logging.config

load_dotenv()

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] [%(module)s:%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "app.log"
        }
    },
    "loggers": {
        "waffles_logger": {
            "level": "INFO",
            "handlers": ["file"]
        }
    }
}

logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger("waffles_logger")

default_cypher = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def showGraph(cypher: str = default_cypher):
    # create a neo4j session to run queries
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run(cypher).graph())
    widget.node_label_mapping = 'id'
    #display(widget)
    return widget

def extract_text_from_pdf(file_name):
    loader = PyPDFLoader(file_name)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)
    return docs

def process_and_store_text(file_name, llm_transformer, graph):
    documents = extract_text_from_pdf(file_name)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


def handle_click(event):
    if not event:
        return
    if pdf_input.value is not None:
        hp.logger.info(f"Processing pdf {str(pdf_input)}")
        pdf_input.save("test.pdf")
        process_and_store_text("test.pdf", llm_transformer, graph)
        hp.logger.info(f"Finished processing pdf")


graph = hp.get_graph()
llm = hp.get_llm()
llm_transformer = LLMGraphTransformer(llm=llm)
chat_history = []


def handle_question(contents, user, instance):
    logger.info("A question is asked: " + contents)
    answer = hp.invoke_chain(contents, chat_history)
    chat_history.append((contents, answer))
    return answer




pdf_input = pn.widgets.FileInput(name="PDF File", accept=".pdf")

button = pn.widgets.Button(name='Submit PDF', button_type='primary')





pn.bind(handle_click, button, watch=True)


chat_interface = pn.chat.ChatInterface(
    callback=handle_question,
    show_clear=False,
    show_undo=False,
    show_button_name=False,
    message_params=dict(
        show_reaction_icons=False,
        show_copy_icon=False,
    ),
    height=700,
    callback_exception="verbose",
    widgets=[pn.chat.ChatAreaInput(placeholder="Enter some text to get a count!")],
)

main = [chat_interface]

template = pn.template.FastListTemplate(
    title="Know about terrorism",
    main=main,
    sidebar=["## Upload PDF", pdf_input, button],
    main_layout=None,
    accent_base_color="#fd7000",
    header_background="#fd7000",
)
template.servable()