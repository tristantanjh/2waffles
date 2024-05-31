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


graph = Neo4jGraph()
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, object, location, or event entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting person, object, location, or event entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
    """
    return final_data

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}

Provide a comprehensive response that includes the following:
1. Identify the key points or events related to the question.
2. Elaborate on each point by providing relevant details, such as dates, locations, and key individuals or groups involved.
3. If there are dates to be included in your answer, convert any timestamps to actual date formats (YYYY-MM-DD) before including them in the answer.
4. Discuss the background information, motives, or reasons behind the events, if available.
5. Use natural language and aim for a well-structured, coherent response that flows logically from one point to another.
6. If there is insufficient information to provide a complete answer, acknowledge the limitations and request the user to upload a PDF document of an article regarding the topic. Inform them that with the additional information from the PDF, you will be able to provide a more comprehensive answer. 

Do not mention "structured data", "provided context", "context" in your answer. Replace those terms with "my dataset".
Give your answer in prose, and avoid bullet points or lists in your response. The answer should be detailed and in paragraph form, providing a comprehensive explanation of the topic based on the context provided.

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

def get_graph():
    return graph

def get_llm():
    return llm

def invoke_chain(question: str, chat_history):
    logger.info("Question asked: " + question)
    logger.info(f"Chat history: " + str(chat_history))
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    if chat_history:
        return chain.invoke(
            {
                "question": question,
                "chat_history": chat_history
            }
        )
    else:
        return chain.invoke(
            {
                "question": question,
            }
        )
# class Rag:
#
#     def __init__(self, graph, llm):
#         self.graph = graph
#         self.llm = llm
#         self.vector_index = Neo4jVector.from_existing_graph(
#             OpenAIEmbeddings(),
#             search_type="hybrid",
#             node_label="Document",
#             text_node_properties=["text"],
#             embedding_node_property="embedding"
#         )
#         self.entity = None
#
#     def handleQuestion(self, question):
#         self.graph.query(
#             "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
#
#         # Extract entities from text
#         class Entities(BaseModel):
#             """Identifying information about entities."""
#
#             names: List[str] = Field(
#                 ...,
#                 description="All the person, organization, or business entities that "
#                 "appear in the text",
#             )
#
#         prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "You are extracting organization and person entities from the text.",
#                 ),
#                 (
#                     "human",
#                     "Use the given format to extract information from the following "
#                     "input: {question}",
#                 ),
#             ]
#         )
#
#         self.entity_chain = prompt | self.llm.with_structured_output(Entities)
#
#     def generate_full_text_query(self, input: str) -> str:
#         """
#         Generate a full-text search query for a given input string.
#
#         This function constructs a query string suitable for a full-text search.
#         It processes the input string by splitting it into words and appending a
#         similarity threshold (~2 changed characters) to each word, then combines
#         them using the AND operator. Useful for mapping entities from user questions
#         to database values, and allows for some misspelings.
#         """
#         full_text_query = ""
#         words = [el for el in remove_lucene_chars(input).split() if el]
#         for word in words[:-1]:
#             full_text_query += f" {word}~2 AND"
#         full_text_query += f" {words[-1]}~2"
#         return full_text_query.strip()
#
#     # Fulltext index query
#     def structured_retriever(self, question: str) -> str:
#         """
#         Collects the neighborhood of entities mentioned
#         in the question
#         """
#         result = ""
#         entities = self.entity_chain.invoke({"question": question})
#         for entity in entities.names:
#             response = self.graph.query(
#                 """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
#                 YIELD node,score
#                 CALL {
#                   WITH node
#                   MATCH (node)-[r:!MENTIONS]->(neighbor)
#                   RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
#                   UNION ALL
#                   WITH node
#                   MATCH (node)<-[r:!MENTIONS]-(neighbor)
#                   RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
#                 }
#                 RETURN output LIMIT 50
#                 """,
#                 {"query": self.generate_full_text_query(entity)},
#             )
#             result += "\n".join([el['output'] for el in response])
#         return result
#
#     def retriever(self, question: str):
#         print(f"Search query: {question}")
#         structured_data = self.structured_retriever(question)
#         unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
#         final_data = f"""Structured data:
#             {structured_data}
#             Unstructured data:
#             {"#Document ".join(unstructured_data)}
#         """
#         return final_data


