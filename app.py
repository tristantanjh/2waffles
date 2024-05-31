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

# def extract_text_from_pdf(file_name):
#     loader = PyPDFLoader(file_name)
#     documents = loader.load()
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
#     docs = text_splitter.split_documents(documents)
#     return docs

# def process_and_store_text(file_name, llm_transformer, graph):
#     documents = extract_text_from_pdf(file_name)
#     graph_documents = llm_transformer.convert_to_graph_documents(documents)
#     graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

def handle_click(event):
    if not event:
        return
    if pdf_input.value is not None:
        hp.logger.info(f"Processing pdf {str(pdf_input)}")
        pdf_input.save("test.pdf")
        text = extract_text_from_pdf("test.pdf")
        clean_text = generate_body_text(text, llm)
        process_and_store_text(clean_text, llm_transformer, graph)
        hp.logger.info(f"Finished processing pdf")

def extract_text_from_pdf(file_name):
    text = ""
    doc = fitz.open(file_name)
    for page_num in range(len(doc)):
        # Load the page
        page = doc.load_page(page_num)
        
        # Convert the PDF page to an image
        image_data = convert_page_to_image(page)
        
        # Convert image data back to a PIL Image object
        image = Image.open(io.BytesIO(image_data))
        
        # Perform OCR using pytesseract on the image and add to text
        ocr_text = pytesseract.image_to_string(image)
        text += ocr_text
        
    return text

def convert_page_to_image(page):
    # Get the pixmap of the page as bytes
    pixmap = page.get_pixmap()
    img_bytes = pixmap.tobytes()
    
    return img_bytes

def process_and_store_text(clean_text, llm_transformer, graph):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    texts = text_splitter.split_text(clean_text)
    documents = [Document(page_content=text) for text in texts]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
    
def generate_body_text(pdf_text, llm):
    prompt = f"""
    You will be given text extracted from a news article PDF.
    The text will contain the main body content, as well as irrelevant sections such as headers, trending news headlines, random metadata, slogans, or the news outlet name. 
    Your task is to extract only the relevant body text from the article, excluding all other irrelevant information.
    To guide you, the body text typically:
    - Is written in paragraph form, with multiple sentences forming a coherent narrative
    - Does not contain short, fragmented phrases or single-sentence headlines
    - Does not include the news outlet name, slogans, or metadata
    - May include quotes or attributions to sources within the paragraphs
    Please output only the extracted body text, without any additional formatting or comments.

    Extracted PDF Text:
    {pdf_text}
    """
    response = llm.invoke(prompt)
    
    print(response.content)
    
    clean_text = re.sub(r'\n', '', response.content)
    return clean_text

def process_and_store_text(clean_text, llm_transformer, graph):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    texts = text_splitter.split_text(clean_text)
    documents = [Document(page_content=text) for text in texts]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

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