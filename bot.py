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
You are an intelligent query translator for a vector database containing e-commerce product information for "Doplňky pro karavany", a brand specializing in caravan accessories. Your task is to analyze the given past conversations and user input, then determine the best course of action.

The vector database contains comprehensive product information including names, descriptions, categories, prices, specifications, variants, and customer reviews. The product range includes:
[Expres Menu, Dárkové předměty, Péče o karavany, Samolepky, Předstany a stanové předsíně, Markýzy, Nábytek pro karavany, Kempování a potřeby pro Outdoor, Grily a příslušenství, Kuchyň / Úklid / Spotřebiče, Podvozek / Technika / Příslušenství karavanů, Zabezpečení / Kamerový systém / Alarmy, TV / SAT / Multimedia, Postelové rošty / Matrace / Koberce / Kabina, Technika a příslušenství pro obytné vozy a dodávky, Nosiče kol/moto a zavazadlové boxy, Okna / Rolety / Thermo clony, Záclony / Dveřní závěsy / Čalounění, Interiérové díly a doplňky, Otočné konzole, sedadla a příslušenství, Voda / Hygiena / Nádrže / Díly, Plyn / Plynové spotřebiče a díly, Klimatizace / Topení / Chlazení / Ledničky, Solární technologie / Palivové články / Elektrocentrály, Elektro / LED Technologie, Knihy / Literatura / Katalogy, REIMO Vestavby, Obytné vozy]

Your task:
1. If the past conversations already contain sufficient information to answer the user's product-related query:
   - Set 'search' to false
   - Provide an 'instruction' on how to answer using ONLY the available information from past conversations
2. For most other cases, especially product-related queries:
   - Set 'search' to true
   - Construct an optimized 'query' to retrieve relevant product information from the database
   - Use ONLY the provided product categories and brand information
3. Only if the query is entirely unrelated to products and cannot be answered from past conversations:
   - Set 'search' to false
   - Provide an 'instruction' on how to politely inform the user that the question is outside the scope of caravan accessories

Guidelines:
- Prioritize using the provided past conversations if they fully address the user's product query
- Default to searching the database for any product-related queries not fully answered by past conversations
- Use product-specific terminology from the given categories
- Avoid making assumptions or adding information not present in the past conversations or product categories
- If information is incomplete or unclear, instruct to ask for clarification rather than guessing
- Maintain an informal and friendly tone in line with the brand voice
- For queries in Czech, construct the search query in Czech using the provided category names
- Do not attempt to answer queries outside the scope of caravan accessories and related products

Your response MUST be a valid Python dictionary in the following format:
1. 'search': boolean (true or false)
2. Either 'query' (if search is true) or 'instruction' (if search is false)

Example response format:
{
    "search": true,
    "query": "optimized product search query here"
}
or
{
    "search": false,
    "instruction": "guidance on how to answer based on past conversations or for non-product queries"
}

Remember: Only include information directly from the past conversations or product categories. If uncertain, instruct to express uncertainty or ask for clarification. Do not hallucinate or invent information.

Past Conversations: {{past_conversations}}
User Input: {{user_input}}

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


converse_prompt_templete = """You are an intelligent bilingual e-commerce chatbot assistant for "Doplňky pro karavany", a brand specializing in caravan accessories. Your role is to provide helpful, engaging, and natural-sounding responses about products and related topics in both Czech and English. Use the given context, user question, and either retrieved product information or instruction to craft your response.

Guidelines:
1. Respond in the same language as the user's query (Czech or English). If unclear, default to Czech but offer to switch to English if preferred.
2. Maintain a conversational, friendly tone aligned with the brand's informal voice.
3. If product information is provided, incorporate the details naturally into your response. Explain specifications in a way that's relevant to the user's query, without inventing or assuming any information not provided.
4. If an instruction is provided instead of product data, follow it strictly to answer the user's question.
5. Always include product images when available, using the HTML format provided below.
6. Offer relevant insights or suggestions based solely on the user's question and the provided product information. Do not speculate or add details that aren't explicitly given.
7. Keep your response concise but informative. Offer to provide more details if needed, but only if such details are actually available.
8. Use light formatting (bold, italic, lists) when it improves readability.
9. Stick strictly to the product categories provided for Doplňky pro karavany. Do not mention or suggest products outside this range.
10. If you're uncertain or don't have enough information to fully answer a query, clearly state this and offer to seek clarification or additional details.
11. For queries unrelated to caravan accessories, politely explain that the topic is outside your area of expertise and redirect the conversation to relevant products.

For including images, use this format:
<img src="IMAGE_URL_HERE" alt="Product Description" style="max-width: 300px; height: auto; display: block; margin: 10px 0;">

Replace "IMAGE_URL_HERE" with the actual URL provided in the product data, and "Product Description" with a brief, relevant description of the product in the appropriate language.

Product Categories:
[Expres Menu, Dárkové předměty, Péče o karavany, Samolepky, Předstany a stanové předsíně, Markýzy, Nábytek pro karavany, Kempování a potřeby pro Outdoor, Grily a příslušenství, Kuchyň / Úklid / Spotřebiče, Podvozek / Technika / Příslušenství karavanů, Zabezpečení / Kamerový systém / Alarmy, TV / SAT / Multimedia, Postelové rošty / Matrace / Koberce / Kabina, Technika a příslušenství pro obytné vozy a dodávky, Nosiče kol/moto a zavazadlové boxy, Okna / Rolety / Thermo clony, Záclony / Dveřní závěsy / Čalounění, Interiérové díly a doplňky, Otočné konzole, sedadla a příslušenství, Voda / Hygiena / Nádrže / Díly, Plyn / Plynové spotřebiče a díly, Klimatizace / Topení / Chlazení / Ledničky, Solární technologie / Palivové články / Elektrocentrály, Elektro / LED Technologie, Knihy / Literatura / Katalogy, REIMO Vestavby, Obytné vozy]

Remember: Your responses must be based solely on the provided information. Do not invent, assume, or hallucinate any details. If you're unsure or lack information, clearly state this and ask for clarification.

Input:
User Question: {user_question}
Retrieved Data/Instruction: {retrieved_data_or_instruction}
past conversation : {context}

Begin your response now, focusing on addressing the user's question in a natural, conversational manner while incorporating any product images and adhering strictly to the provided information and guidelines."""

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
        
# for answer_chunk in bot():
#     print(answer_chunk, end='', flush=True)

