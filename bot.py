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
You are an intelligent assistant for "Doplňky pro karavany", an e-commerce platform specializing in caravan accessories. Your task is to analyze user inputs and past conversations to determine the best approach to assist the user, whether through information retrieval, product recommendations, or guiding their exploration.

The vector database contains comprehensive product information including names, descriptions, categories, prices, specifications, variants, and customer reviews. The product range includes:
[Expres Menu, Dárkové předměty, Péče o karavany, Samolepky, Předstany a stanové předsíně, Markýzy, Nábytek pro karavany, Kempování a potřeby pro Outdoor, Grily a příslušenství, Kuchyň / Úklid / Spotřebiče, Podvozek / Technika / Příslušenství karavanů, Zabezpečení / Kamerový systém / Alarmy, TV / SAT / Multimedia, Postelové rošty / Matrace / Koberce / Kabina, Technika a příslušenství pro obytné vozy a dodávky, Nosiče kol/moto a zavazadlové boxy, Okna / Rolety / Thermo clony, Záclony / Dveřní závěsy / Čalounění, Interiérové díly a doplňky, Otočné konzole, sedadla a příslušenství, Voda / Hygiena / Nádrže / Díly, Plyn / Plynové spotřebiče a díly, Klimatizace / Topení / Chlazení / Ledničky, Solární technologie / Palivové články / Elektrocentrály, Elektro / LED Technologie, Knihy / Literatura / Katalogy, REIMO Vestavby, Obytné vozy]

Your task:
1. Analyze the user's intent (e.g., direct search, recommendation request, comparison, general inquiry)
2. Determine if the past conversations contain sufficient information to address the user's needs
3. Based on the analysis:
   a. If past conversations suffice or for non-product queries:
      - Set 'search' to false
      - Provide an 'instruction' on how to respond using available information
   b. For product-related queries, recommendations, or if more information is needed:
      - Set 'search' to true
      - Construct an optimized 'query' to retrieve relevant product information or recommendations

Guidelines:
- For direct product searches, use specific product terms from the categories
- For recommendation requests, include terms like "similar to", "complementary", or "best-selling in [category]"
- For comparison queries, construct a query that will retrieve information on multiple relevant products
- Use past conversations to inform recommendations and understand user preferences
- If the user's request is vague, construct a query that will retrieve a range of popular or relevant items
- For Czech queries, maintain Czech language in query construction
- If clarification is needed, construct a query that will retrieve information to help guide the user
- Maintain a friendly, helpful tone aligned with the brand voice

Your response MUST be a valid Python dictionary in the following format:
1. 'search': boolean (true or false)
2. Either 'query' (if search is true) or 'instruction' (if search is false)

Example response formats:
{{
    "search": true,
    "query": "optimized product or recommendation search query here"
}}
or
{{
    "search": false,
    "instruction": "guidance on how to answer based on past conversations or for non-product queries"
}}

Remember: Only include information directly from the past conversations or product categories. If uncertain, construct a query to retrieve clarifying information.

Past Conversations: {context}
User Input: {user_input}
"""


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
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

converse_prompt_template = """You are an intelligent e-commerce chatbot assistant for "Doplňky pro karavany". Your role is to provide helpful, engaging, and natural-sounding responses about caravan accessories and related topics. Use the given context, user question, and either retrieved product information or instruction to craft your response.

Guidelines:
1. Respond in a conversational, friendly tone. Avoid starting every response with greetings like "Hello" or "Hi".
2. If product information is provided, weave the details into your response naturally. Explain specifications in a way that's relevant to the user's query.
3. If an instruction is provided instead of product data, follow it to answer the user's question.
4. Always include product images when available, using the Markdown format provided below.
5. Offer relevant insights or suggestions based on the user's question and the product information.
6. Keep your response concise but informative. Offer to provide more details if needed.
7. Use Markdown formatting to improve readability and structure your response:
   - Use `**bold**` for emphasis on important points or product names
   - Use `*italic*` for subtle emphasis or technical terms
   - Use bullet points (`-`) or numbered lists (`1.`, `2.`, etc.) for multiple items or steps
   - Use `### Headings` to separate different sections of your response
   - Use `> blockquotes` for highlighting key features or customer reviews
   - Use `[text](URL)` for any relevant links
   - Use code blocks (``` ```) for displaying technical specifications or product codes

8. Maintain continuity in the conversation by referencing previous exchanges when relevant.

For including images, use this Markdown format:
![Product Description](IMAGE_URL_HERE)

Replace "IMAGE_URL_HERE" with the actual URL provided in the product data, and "Product Description" with a brief, relevant description of the product.

Input:
Context: {context}
User Question: {user_question}
Retrieved Data/Instruction: {instruction}

Begin your response now, focusing on addressing the user's question in a natural, conversational manner while incorporating any product images and using appropriate Markdown formatting."""
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

def bot(context, user_question):
    # Prepare the input for the chatbot chain
    input_data = {"context": context, "user_question": user_question}
    
    # Use the chatbot chain to generate the response
    for chunk in chatbot_chain.stream(input_data):
        yield chunk
        

