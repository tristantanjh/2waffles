from bs4 import BeautifulSoup as soup
import requests
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def scrape_wiki():
    base_url = "https://en.wikipedia.org/wiki/List_of_terrorist_incidents_in_"
    # wiki_url = "https://en.wikipedia.org/wiki/List_of_terrorist_incidents"
    for i in range(2001, 2024):
        # print(i)

        # html = requests.get(base_url + str(i))
        # bsobj = soup(html.content, 'lxml')
        # headers = bsobj.findAll("td", class_=False)
        # # headers = bsobj.find("b", text=lambda text: text and text.startswith('List of terrorist incidents in '))
        # headers = [header for header in headers if header.find('b') and header.find('b').find('a')]        # print(headers)
        # incidents = [header.b.a.text for header in headers]

        graph = Neo4jGraph(url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"],
                           password=os.environ["NEO4J_PASSWORD"])
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")  # gpt-4-0125-preview occasionally has issues
        llm_transformer = LLMGraphTransformer(llm=llm)
        raw_documents = WikipediaLoader(query="September 11 attacks").load()
        print(raw_documents)
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        documents = text_splitter.split_documents(raw_documents[:3])
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        break
        # for incident in tqdm(incidents):
        #     print(incident)
        #     raw_documents = WikipediaLoader(query=incident).load()
        #     print(raw_documents)
        #     # text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        #     # documents = text_splitter.split_documents(raw_documents[:3])
        #     # graph_documents = llm_transformer.convert_to_graph_documents(documents)
        #     # graph.add_graph_documents(
        #     #     graph_documents,
        #     #     baseEntityLabel=True,
        #     #     include_source=True
        #     # )
        #     break
        # break
        # count = 0
        # print(headers[-1].b.a.text)
        # for header in headers:
        #     print(header.b.a.text)
        #     count += 1
        #     if count == 5:
        #         break
        # break

scrape_wiki()