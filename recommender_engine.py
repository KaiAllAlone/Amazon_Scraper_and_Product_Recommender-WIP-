import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import re

def format_response(text):
    # Adds newlines only between recommendations (reasoning followed by numbered line)
    # Matches: <reasoning>\d. <title>...
    text = re.sub(r"(\\S.*?)\\s*(?=\\d+\\.\\s[A-Z])", r"\\1\\n", text)
    return text.strip()

load_dotenv()

# Globals to persist across requests
qa_chain = None
user_pref_summary = None

def initialize_documents(path, chunk_size=1000, chunk_overlap=50):
    items = json.load(open(path, 'r', encoding='utf-8'))
    raw_docs = []
    for item in items:
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
        raw_docs.append(Document(page_content=content, metadata=item))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(raw_docs)

async def setup_engine():
    global qa_chain, user_pref_summary

    print("[INFO] Loading and splitting product documents...")
    docs = initialize_documents('electronics_gaming_products.json')

    print(f"[INFO] Embedding {len(docs)} documents...")
    embed_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectorstore = FAISS.from_texts(texts, embed_model, metadatas=metadatas)
    retriever = vectorstore.as_retriever()

    print("[INFO] Generating user preference summary...")
    orders = json.load(open('orders.json', 'r', encoding='utf-8'))
    product_list = ''.join([f"Product:{o['title']},Price:{o['price']}\n" for o in orders])
    llm_pref = ChatOpenAI(
        model='llama3-70b-8192',
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.7
    )
    prompt = PromptTemplate.from_template(
        """
        You are a smart assistant analyzing user shopping behavior.
        Below are items the user has purchased:\n{products}\nSummarize preferences into 3-4 concise sentences.
        """
    )
    user_pref_summary = llm_pref.invoke(prompt.format(products=product_list)).content.strip()
    print(f"[READY] Preference summary: {user_pref_summary}")

    print("[INFO] Setting up memory and LLM chain...")
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True, k=3)
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        temperature=0.5
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    print("[READY] Recommender engine initialized.")

async def ask_bot(user_input: str) -> str:
    global qa_chain, user_pref_summary
    if qa_chain is None:
        raise RuntimeError("Engine not initialized. Call setup_engine() first.")
    question = f"""
    [SYSTEM MESSAGE]
You are a product recommendation bot that must follow this format EXACTLY.

USER PREFERENCES:
{user_pref_summary}
You must always reply in the following format:

1. Logitech MX Master 3S Available:-Yes Rating:-4.8 Brand:-Logitech Price:-₹8499
Perfect for professionals needing an ergonomic and silent productivity mouse.

2. Sony WH-1000XM5 Available:-Yes Rating:-4.7 Brand:-Sony Price:-₹29999
Top-tier noise cancellation and long battery life for frequent travelers.

3. JBL Flip 5 Available:-No Rating:-4.3 Brand:-JBL Price:-₹8499
Loud, punchy Bluetooth speaker ideal for outdoor use.

NO introduction. NO conclusion. NO explanations outside this structure.

If no recommendations are found, respond with exactly:
No recommendations found

If the user asks to recommend without mentioning product specifics use {user_pref_summary} to generate recommendations.

[USER QUESTION]
{user_input}
"""


    result = qa_chain.invoke({"question": question})
    raw_answer = result.get('answer', 'Sorry, no answer.')
    return format_response(raw_answer)
