import requests
from langchain_chroma import Chroma
import xml.etree.ElementTree as ET
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import ast
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import chain, RunnableLambda
import json

import os

api_key = os.getenv('api_key')


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

vector_store = Chroma(
    collection_name="Product",
    embedding_function=embeddings,
    persist_directory="./",  # Where to save data locally, remove if not neccesary
)


retrieval_prompt_template = """
You are an intelligent query translator for a vector database containing e-commerce product information. Your task is to analyze the given context and user input, then determine the best course of action.

The vector database contains comprehensive product information including names, descriptions, categories, prices, specifications, variants, and customer reviews.

Your task:

1. If the context already contains sufficient information to answer the user's product-related query:
   - Set 'search' to false
   - Provide an 'instruction' on how to answer using the available context

2. For most other cases, especially product-related queries:
   - Set 'search' to true
   - Construct an optimized 'query' to retrieve relevant product information from the database

3. Only if the query is entirely unrelated to products and cannot be answered from context:
   - Set 'search' to false
   - Provide an 'instruction' on how to answer the non-product-related question

Guidelines:
- Prioritize using the provided context if it fully addresses the user's product query
- Default to searching the database for any product-related queries not fully answered by the context
- Consider product-specific terminology and various phrasings of customer inquiries
- Only avoid searching if the query is completely unrelated to products/e-commerce and not answered by context

Your response should be a Python dictionary:
1. 'search': boolean (true or false)
2. Either 'query' (if search is true) or 'instruction' (if search is false)

Example response format:
{{
    "search": true,
    "query": "optimized product search query here"
}}

or
{{
    "search": false,
    "instruction": "guidance on how to answer based on context or for non-product queries"
}}
Context:{context}
User Input: {user_input}

"""


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


retrieve_prompt = PromptTemplate(
    template=retrieval_prompt_template, input_variables=["context", "user_input"]
)
parser = StrOutputParser()

retreive_chain = retrieve_prompt | llm | parser


def retreive(input):
    print(input)
    context = input["context"]
    user_question = input["user_question"]

    answer = retreive_chain.invoke({"context": context, "user_input": user_question})
    cleaned_json = re.sub(r"^[^{]*|[^}]*$", "", answer)
    print(cleaned_json)

    query = json.loads(cleaned_json)
    if query["search"]:
        search_term = query["query"]
        print(search_term)
        return retriever.invoke(search_term)
    else:
        
        instruction = query["instruction"]
        return instruction


converse_prompt_templete = """You are an intelligent e-commerce chatbot assistant. Your role is to provide helpful, engaging, and natural-sounding responses about products and related topics. Use the given context, user question, and either retrieved product information or instruction to craft your response.

Guidelines:
1. Respond in a conversational, friendly tone.
2. If product information is provided, weave the details into your response naturally. Don't just list specifications; explain them in a way that's relevant to the user's query.
3. If an instruction is provided instead of product data, follow it to answer the user's question.
4. Always include product images when available, using the HTML format provided below.
5. Offer relevant insights or suggestions based on the user's question and the product information.
6. Keep your response concise but informative. Offer to provide more details if needed.
7. Use light formatting (bold, italic, lists) when it improves readability.

For including images, use this format:
<img src="IMAGE_URL_HERE" alt="Product Description" style="max-width: 300px; height: auto; display: block; margin: 10px 0;">

Replace "IMAGE_URL_HERE" with the actual URL provided in the product data, and "Product Description" with a brief, relevant description of the product.

Input:
Context: {context}
User Question: {user_question}
Retrieved Data/Instruction: {instruction}

Begin your response now, focusing on addressing the user's question in a natural, conversational manner while incorporating any product images."""

converse_prompt = PromptTemplate(
    template=converse_prompt_templete,
    input_variables=["context", "user_question", "instruction"],
)
converse_chain = converse_prompt | llm | parser


chatbot_chain = (
    {
        "instruction": RunnableLambda(retreive),
        "user_question": RunnablePassthrough(),
        "context": RunnablePassthrough(),
    }
    | converse_chain
    | parser
)

def bot(context='hi', user_question='what can you do'):
    # Prepare the input for the chatbot chain
    input_data = {"context": context, "user_question": user_question}
    
    # Use the chatbot chain to generate the response
    for chunk in chatbot_chain.stream(input_data):
        yield chunk
        
for answer_chunk in bot():
    print(answer_chunk, end='', flush=True)

