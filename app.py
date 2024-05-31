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
# from yfiles_jupyter_graphs import GraphWidget
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

def extract_text_from_pdf(file_name):
    text = ""
    doc = fitz.open(file_name)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_data = convert_page_to_image(page)
        image = Image.open(io.BytesIO(image_data))
        ocr_text = pytesseract.image_to_string(image)
        text += ocr_text
    return text


def convert_page_to_image(page):
    pixmap = page.get_pixmap()
    img_bytes = pixmap.tobytes()
    return img_bytes


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
    hp.update_pdf_context(response.content)
    clean_text = re.sub(r'\n', '', response.content)
    return clean_text


def process_and_store_text(clean_text, llm_transformer, graph):
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    texts = text_splitter.split_text(clean_text)
    documents = [Document(page_content=text) for text in texts]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)


def handle_click(event):
    if not event:
        return
    if pdf_input.value is not None:
        spinner.value = True
        spinner.name = "Uploading PDF..."
        chat_area_input.disabled = True
        hp.logger.info(f"Processing pdf {str(pdf_input)}")
        pdf_input.save("test.pdf")
        text = extract_text_from_pdf("test.pdf")
        clean_text = generate_body_text(text, llm)
        process_and_store_text(clean_text, llm_transformer, graph)
        hp.logger.info(f"Finished processing pdf")
        chat_area_input.disabled = False
        spinner.name = ""
        spinner.value = False

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
    global chat_history
    logger.info("A question is asked: " + contents)
    local_agent = hp.build_workflow()
    answer = hp.run_agent(contents, local_agent, chat_history)
    chat_history.append((contents, answer))
    return answer

pdf_input = pn.widgets.FileInput(name="PDF File", accept=".pdf")

button = pn.widgets.Button(name='Submit PDF', button_type='primary')

spinner = pn.indicators.LoadingSpinner(value=False, name="", size=30)



pn.bind(handle_click, button, watch=True)

chat_area_input = pn.chat.ChatAreaInput(placeholder="Ask me anything about terrorism.", disabled=False)

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
    widgets=[chat_area_input],
)

main = [chat_interface]

template = pn.template.FastListTemplate(
    title='Unmasking terrorism',
    main=main,
    sidebar=["## Upload PDF", pdf_input, button, spinner],
    main_layout=None,
    accent_base_color="#6d449e",
    header_background="#6d449e",
    logo="./assets/logo2.png"
)

template.modal.append("## Submited PDF")
template.open_modal()

template.servable()