from scraper_and_best_price_finder import login_and_scrape_amazon_orders as scrape_orders
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from sentence_transformers.util import cos_sim
import numpy as np
import os
import time
import json

load_dotenv()
def convert_json_to_docs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    docs = []
    for item in data:
        title = str(item.get("title", "")).strip()
        desc = str(item.get("description", "")).strip()
        price = f"₹{item.get('price', 'N/A')}"
        category = str(item.get("category", "")).strip()
        brand= str(item.get("brand", "")).strip()
        rating= str(item.get("rating", "N/A")).strip()
        features= str(",".join(item.get("features", "N/A"))).strip()
        availability = str(item.get("availability", "N/A")).strip()
        content = (
            f"Title: {title}\n"
            f"Description: {desc}\n"
            f"Price: {price}\n"
            f"Category: {category}\n"
            f"Brand: {brand}\n"
            f"Rating: {rating}\n"
            f"Features: {features}\n"
            f"Availability: {availability}\n"

        )

        docs.append(Document(page_content=content, metadata={"product_id": item.get("product_id", "")}))
    return docs
# scrape_orders()  # Scrape Amazon orders and save to orders.json

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def describe_preferences(_:str)->str:
    with open("orders.json", "r") as f:
        orders = json.load(f)
    products=""
    for order in orders:
        products+=f"Product:{order['title']},Price: {order['price']}\n"
    llm=ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    prompt= PromptTemplate.from_template(""" You are a smart assistant analyzing user shopping behavior. Below are items the user has purchased:

                                            {products}

                                            Based on this:
                                            - Group products by common categories and brands.
                                            - Identify typical price ranges.
                                            - List any repeated features or specs (e.g., wireless, gaming, ergonomic).
                                            - Summarize these preferences into 3-4 concise sentences, focusing on practical shopping intent (e.g., "The user prefers mid-range gaming accessories with ergonomic designs and wireless features").

                                            Do not list products. Just summarize the **pattern** and **preferences**.""")
    formatted_prompt=prompt.format(products=products)
    result=llm.invoke(formatted_prompt)
    return result.content.strip()      

def recommend_from_document(docs,user_preferences):
    prompt_string = f"""
                    Here are the user's preferences:

                    {user_preferences.strip()}
                    Based on this, recommend up to 5 relevant products from the given context.
                    Include both close and loose matches if they seem practically useful.
                    Prioritize close matches, but include slightly higher-priced or feature-rich alternatives if they provide better value or match user intent.
                    IMPORTANT:- If user specifically looks for adjectives like 'expensive','high-end' etc. make those preferences the highest priority above all the previous ones
                    Each recommendation must be on a separate line, and clearly mention:
                    - Title
                    - Availability
                    - Rating
                    - Brand
                    - Price
                    The recommendations must follow this format:-
                    <numbering> <title> Available:-<availability> Rating:-<rating> Brand:-<brand> Price:-<price>
                    <reasoning>
                    (Add a very short 1-line reasoning if helpful.)
                    NO EXTRA REASONINGS OR TEXT OUTSIDE FORMAT.
                    If no good matches exist at all, return "No recommendations found".
                    """

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts, metadatas = [], []
    for i, doc in enumerate(docs):
        content = getattr(doc, "page_content", None)

        if not isinstance(content, str):
            print(f"[SKIPPED] doc[{i}] - Not a string: {type(content)}")
            continue

        content = content.strip()
        if not content:
            print(f"[SKIPPED] doc[{i}] - Empty string after stripping")
            continue

        texts.append(content)
        metadatas.append(doc.metadata if isinstance(doc.metadata, dict) else {})
    print(f"[INFO] Total documents to embed: {len(texts)}")
    if not texts:
        print("[ERROR] No valid texts found for embedding. Exiting.")
        return
    vectorstore = FAISS.from_texts(texts,embedding_model,metadatas=metadatas)
    retriever=vectorstore.as_retriever()
   
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True,
        k=2
    )
    qa_chain=ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5
    ),chain_type="stuff",retriever=retriever,verbose=False,memory=memory)
    print("Starting conversation with the bot. Type 'n' to exit.")
    query=prompt_string
    while(True):
        result=qa_chain.invoke({"question":query})
        print(f"Bot:-{result['answer']}")
        c=(input("You:-")).strip().lower()
        if(c=='n'):
            print("Recommendation session ended.")
            break;
        query=c+f"keep in mind {prompt_string}"
    return
def run_recommendation():
    docs = convert_json_to_docs("electronics_gaming_products.json")
    docs = split_documents(docs, chunk_size=1000, chunk_overlap=50)
    user_preferences = describe_preferences(" ")

    # Instead of CLI, return the recommendation result
    return recommend_from_document(docs, user_preferences)

if __name__=='__main__':
    docs=convert_json_to_docs(r"electronics_gaming_products.json")
    docs=split_documents(docs,chunk_size=1000, chunk_overlap=50)
    user_preferences=describe_preferences(" ")
    recommend_from_document(docs,user_preferences)
        
    
    


    

                                         
    