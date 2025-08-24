from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import json
import tqdm
from serpapi import GoogleSearch
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os
load_dotenv()

BRAVE_PATH = os.getenv("BRAVE_PATH", "/usr/bin/brave-browser")
chromedriver_path = os.getenv("chromedriver_path", "/usr/local/bin/chromedriver")
def login_and_scrape_amazon_orders():
    options=Options()
    options.binary_location=BRAVE_PATH
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(chromedriver_path), options=options)
    driver.get("https://www.amazon.in")  
    print("Please log in to your Amazon account in the opened browser window.")
    # Wait for user to log in
    time.sleep(20) 
    # Log in to Amazon
    driver.get("https://www.amazon.in/gp/cart/view.html?ref_=nav_cart")
    time.sleep(5)  # Let page load
    orders = []
    #  scraping the wishlist
    order_cards = driver.find_elements(By.XPATH,'//div[contains(@class, "sc-list-item") and @role="listitem"]')
    for card in order_cards:
        try:
            title_elem=card.find_element(By.XPATH,'.//span[@class="a-truncate-cut"]').text.strip()
        except:
            title_elem="Title not found"
        try:
            order_price=card.find_element(By.XPATH,'.//span[@class="a-price-whole"]').text.strip()
        except:
            order_price="Price not available"
        orders.append({
            "title":title_elem,
            "price":order_price
        })
    #scraping the past orders
    driver.get("https://www.amazon.in/gp/css/order-history?ref_=nav_orders_first")
    time.sleep(5)  # Let page load
    order_cards = driver.find_elements(By.XPATH,'//li[contains(@class, "order-card__list")]')
    for card in order_cards:
        try:
            title_elem=card.find_element(By.XPATH,'.//div[@class="yohtmlc-product-title"]').text.strip()
        except:
            title_elem="Title not found"
        try:
            order_price=card.find_element(By.XPATH,'.//span[contains(text(), "Total")]/../../div[2]/span').text.strip()
        except:
            order_price="Price not available"
        orders.append({
            "title": title_elem,
            "price": order_price
        })
    driver.quit()
    with open("orders.json","w",encoding="utf-8") as f:
        json.dump(orders,f,indent=4,ensure_ascii=False)
    return orders






def compare_prices_with_langchain(orders):
    from dotenv import load_dotenv
    load_dotenv()
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.2
    )
    
    search = SerpAPIWrapper()
    
    # Load orders from JSON file or scrape them
    for order in orders:
        print(f"\nSearching for: {order['title']}")

        # Perform actual search using SerpAPI
        query=f"{order['title']} price Flipkart Snapdeal Amazon India"
        results=search.results(query)

        search_snippets=[]
        if 'organic_results' in results:
            for res in results['organic_results']:
                title=res.get('title', '')
                link=res.get('link', '')
                snippet=res.get('snippet', '')
                search_snippets.append(f"- {title}\n  {snippet}\n  🔗 {link}")
        joined_results="\n".join(search_snippets)
        
        
        
        # LLM prompt
        prompt = f"""
        You are a smart shopping assistant.

        The user has purchased:
        - Product: {order['title']}
        - Amazon Price: ₹{order['price']}
        Here are some search results:
        {joined_results}

        Your job:
        1. Search these results for cheaper alternatives to the above product.
        2. Only report if the price is clearly lower or the reviews are significantly better.
        3. Include:
        - Product name
        - Price
        - Store name
        - Link
        4.You may also recommend alternatives with better reviews or cheaper price.
        5.Be concise and accurate. If nothing better is found, say that.
        6. At the end of each recommendation mention how much cheaper the
           suggested product is.
        7. IMPORTANT- The results' links must always direct to e-commerce websites.
        """



        response = llm.invoke(prompt)
        print(f"Result:\n{response.content}")
        print("-" * 50)
        cont = input("Continue to next order? (y/n): ").strip().lower()
        if cont !='y':
            break
        
if __name__ == "__main__":
    orders = login_and_scrape_amazon_orders()
    compare_prices_with_langchain(orders)