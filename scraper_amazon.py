from scraper_and_best_price_finder import login_and_scrape_amazon_orders as scrape_orders
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.utilities import SerpAPIWrapper
from langchain.tools import Tool
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import os
import time
import json

scrape_orders()
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
        temperature=0.5
    )
    prompt= PromptTemplate.from_template(""" You are a helpful user purchase preferences analyzer.
                                         You are given a list of products purchased by a user:-{products}.
                                         Summarize the user's purchase preferences in a few sentences.""")
    formatted_prompt=prompt.format(products=products)
    result=llm.invoke(formatted_prompt)
    return result.content.strip()

load_dotenv()
llm=ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2
    )
user_pref_desc=describe_preferences("Describe the user's purchase preferences based on their past orders.")
query_prompt=PromptTemplate.from_template("""Generate 5 diverse product search queries that match the user's taste.Given user's preferences: {user_pref_desc}.
                                          1.These queries should be suitable for searching on Google,Amazon or other Indian E-commerce sites.
                                          2.Keep the queries short around 10 words no need for unnecessary details. 
                                          3.Each query should be a single line and the queries must be on seperate lines
                                          4.At the end of each query add "Flipkart Snapdeal Amazon India"
                                          IMPORTANT:-Do not include any additional text or explanations,just the queries.""")
chain = query_prompt | llm
print("Generating queries based on user preferences...")
queries_output = chain.invoke({"user_pref_desc": user_pref_desc}).content.strip()
print("Generated Queries:")
queries=[]
search_snippets = []
search=SerpAPIWrapper()
for query in queries_output.split("\n"):
    if query.strip():
        queries.append(query.strip())
for q in queries:
        time.sleep(2)
        print(f"\n Searching:-{q}")
        results = search.results(q)
        if 'organic_results' in results:
            for res in results['organic_results']:
                title = res.get('title', '')
                link = res.get('link', '')
                snippet = res.get('snippet', '')
                search_snippets.append(f"{title}\n  {snippet}\n  🔗 {link}")
joined_res="\n".join(search_snippets)

model = SentenceTransformer("all-MiniLM-L6-v2")
with open("orders.json", "r") as f:
        orders = json.load(f)
past_purchases=[]
for order in orders:
    past_purchases.append(order['title'] + " " + order['price'])
purchase_embeddings = model.encode(past_purchases)
user_vector = purchase_embeddings.mean(axis=0)
search_results = []
for res in search_snippets:
    title = res.split('\n')[0].replace('-', '')
    snippet = res.split('\n')[1]
    link = res.split('🔗 ')[-1]
    search_results.append((title, snippet, link))
search_embeddings = model.encode([result[0] for result in search_results])
similarities = cos_sim(user_vector, search_embeddings).flatten()
top_indices = np.argsort(similarities)[-5:][-5:]
print("\n Similar Products Found:")
for idx in top_indices:
    title, snippet, link = search_results[idx]
    print(f"- {title}     🔗 {link}\n")
# Save the results to a JSON file
with open("similar_products.json", "w") as f:
    json.dump([search_results[idx] for idx in top_indices], f, indent=4)



    

                                         
    