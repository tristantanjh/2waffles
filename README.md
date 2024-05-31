# LifeHack 2024 - 2 Waffles

## Team Members

- [Tristan Tan](https://github.com/tristantanjh)
- [Oscar Qian](https://github.com/oscarqjh)
- [Joshau Goh](https://github.com/joshuagohez)
- [Qu Zhetao](https://github.com/quzhetao01)

### Website Link: 

### Video Demo Link: 

---

## Table of Contents
- [Project Details](#project-details)
- [Features](#features)
- [How We Did It](#how-we-did-it)
- [What We Learnt](#what-we-learnt)

## Project Details
Reports and articles on terrorism usually contain a lot of text, which are unstructured and difficult to resolve across reports in an automated fashion. For instance, linking articles about a terror incident, all coming in at different times in a day with varying details, can be challenging. Each report contains entities (Person, Object, Location, Events) that could be represented in a knowledge graph and accessed through a chatbot. This project aims to design and implement a Large Language Model (LLM) that can extract entities from reports into a knowledge graph, and an LLM that can answer questions based on the generated knowledge graph.

_Made for NUS' LifeHack 2024 - Theme 3, Subtheme 2._

---

## Features

Our project leverages LLMs (OpenAI) and Neo4j's knowledge graphs to create a comprehensive system for managing and querying information about terrorism incidents. The project consists of several key components:

1. **User Interface**
  - Chatbot: The chatbot interface allows users to ask queries about terrorism. It retrieves relevant information from the Neo4j database and provides responses based on the knowledge graph.
  - PDF Uploader: Users can upload PDF files containing articles about terrorism. The system extracts text from the PDFs, processes the content to identify entities, and updates the knowledge graph accordingly.
2. **Backend Components**
  - Knowledge Graph: A Neo4j-based graph database that stores entities (Person, Object, Location, Events) and their relationships extracted from terrorism reports and articles.
  - Web Scraper: A scheduler-based web scraper that periodically gathers new articles from reputable news sources and updates the knowledge graph.
  - RAG Model: Combines information retrieval and generation to provide accurate and relevant responses to user queries by leveraging both the knowledge graph and external web searches.

## How We Did It

1. Knowledge Graph Prepopulation:
  - Collected data from an online database of known terror attacks.
  - Processed the data to extract entities and relationships.
  - Populated the Neo4j knowledge graph with the preprocessed data.

2. RAG Model Implementation:
  - Integrated a Retrieval-Augmented Generation model to handle user queries.
  - Configured the model to retrieve relevant nodes and relationships from the Neo4j database.
  - Set up the LLM to generate responses based on the retrieved information.

3. Web Scraping Scheduler:
  - Developed a web scraper to collect news articles from AP News and CNA.
  - Implemented a scheduling mechanism to run the scraper weekly.
  - Processed and stored the scraped articles in a MongoDB database, ensuring no duplicates.

4. PDF Processing:
  - Created a function to extract text from uploaded PDFs using Tesseract OCR.
  - Developed an entity extraction pipeline to identify and store relevant entities in the knowledge graph.
  - Allowed users to upload PDFs through the chatbot interface.

5. Web Search Functionality:
  - Configured the LLM to perform web searches if the knowledge graph does not contain sufficient information to answer a user query.
  - Extracted relevant information from the web search results and included it in the chatbot response.

## What We Learnt

Throughout the course of this project, we learned several key lessons:

- Entity Extraction: Extracting structured information from unstructured text is a challenging yet crucial step in building a useful knowledge graph. We learned various techniques for entity extraction and relationship mapping.
- Knowledge Graphs: Working with Neo4j provided insights into the advantages of graph databases, especially for storing and querying complex relationships between entities.
- LLM Integration: Integrating LLMs with knowledge graphs can significantly enhance the ability to retrieve and generate relevant responses, demonstrating the power of combining different AI technologies.
- Web Scraping: Developing a reliable and efficient web scraping system taught us the importance of handling various edge cases and ensuring data quality.
- Scheduling and Automation: Implementing scheduling mechanisms for periodic tasks (like web scraping) highlighted the importance of automation in maintaining an up-to-date knowledge base.


