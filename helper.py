from dotenv import load_dotenv
from typing_extensions import TypedDict
import os
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun, BingSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, BingSearchAPIWrapper
from langgraph.graph import END, StateGraph
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

    result_list = result.split("\n")
    counter = 0
    for entity in entities.names:
        print("Entity:" + entity)
        for i in range(len(result_list)):
            print(result_list[i])
            if entity.lower() in result_list[i].lower():
                counter += 1
    if counter == 0:
        result = ""
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
Use natural language and be as elaborate as possible.
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
    
########################################################### Web Search Tool ###########################################################
wrapper = DuckDuckGoSearchAPIWrapper(max_results=25)
web_search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)

########################################################### Query Transformation ###########################################################
query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are an expert at crafting web search queries for research questions.
            More often than not, a user will ask a basic question that they wish to learn more about, however it might not be in the best format. 
            Reword their query to be the most effective web search string possible.
            Return the JSON with a single key 'query' with no premable or explanation. 
            
            Question to transform: {question} 
         """)
    ]
)

# Chain
query_chain = query_prompt | llm | JsonOutputParser()

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are an AI assistant for Research Question Tasks, that synthesizes web search results. 
            Strictly use the following pieces of web search context to answer the question. If you don't know the answer, just say that you don't know. 
            keep the answer concise, but provide all of the details you can in the form of a research report. 
            Only make direct references to material if provided in the context.
         """),
         ("human", """
            Given the following context, answer the question as best as you can.
            Context: {context}
            Question: {question}
         """)
    ]
)

# Chain
generate_chain = generate_prompt | llm | StrOutputParser()

############################################################# Graph State #############################################################
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search_query: revised question for web search
        context: web_search result
    """
    question : str
    generation : str
    search_query : str
    context : str
    history: List[Tuple[str, str]]

# Node - Generate

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print("Step: Generating Final Response")
    question = state["question"]
    history = state["history"]

    # Answer Generation
    generation = invoke_chain(question, history)
    return {"generation": generation}

def generate_for_web(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    
    print("Step: Generating Final Response")
    question = state["question"]
    context = state["context"]

    # Answer Generation
    generation = generate_chain.invoke({"question": question, "context": context})
    return {"generation": generation}

# Node - Query Transformation

def transform_query(state):
    """
    Transform user question to web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended search query
    """
    
    print("Step: Optimizing Query for Web Search")
    question = state['question']
    gen_query = query_chain.invoke({"question": question})
    search_query = gen_query["query"]
    return {"search_query": search_query}


# Node - Web Search

def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to context
    """

    search_query = state['search_query']
    print(f'Step: Searching the Web for: "{search_query}"')
    
    # Web search tool call
    search_result = web_search_tool.invoke(search_query)
    return {"context": search_result}


# Conditional Edge, Routing

def route_question(state):
    """
    route question to web search or generation.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("Step: Routing Query")
    question = state['question']
    structured_data = structured_retriever(question)

    print(f"Structured Data: {structured_data}")
    
    if len(structured_data) != 0:
        print("Step: Context Found, Routing to Generation")
        return "generate"
    elif len(structured_data) == 0:
        print("Step: Context Not Found, Routing to Web Search")
        return "websearch"
    
def build_workflow():
    """
    Build the workflow for the graph
    """
    # Build the nodes
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", web_search)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_for_web", generate_for_web)

    # Build the edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "websearch")
    workflow.add_edge("websearch", "generate_for_web")
    workflow.add_edge("generate", END)
    workflow.add_edge("generate_for_web", END)

    # Compile the workflow
    local_agent = workflow.compile()

    return local_agent

def run_agent(query, local_agent, chat_history):
    output = local_agent.invoke({"question": query, "history": chat_history})
    return output['generation']


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


