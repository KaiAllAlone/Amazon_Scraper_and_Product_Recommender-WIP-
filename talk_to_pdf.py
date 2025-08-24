import langchain
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import tqdm
import os
load_dotenv()
def pdf_to_doc(path):
    """Convert a PDF file to a list of Document objects."""
    from langchain.document_loaders import PyPDFLoader
    docs=[]
    loader = PdfReader(path)
    pbar=tqdm.tqdm(total=len(loader.pages), desc="Processing PDF pages")
    for page in loader.pages:
        pbar.update(1)
        text = page.extract_text()
        if text:
            docs.append(Document(page_content=text))
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50
     )
    split_docs = splitter.split_documents(docs)
    return split_docs


def vectorize(docs):
    
    embedding_model=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs,embedding_model)
    return vectorstore

def create_qa_agent(vectorstore):
    memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    return_messages=True,
    k=2
)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_base=os.getenv("OPENAI_API_BASE"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3
    ),
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        memory=memory,
    )
    prompt=PromptTemplate(
        input_variables=["question"],
        template="You are a helpful assistant. Answer the question based on the provided documents.\n\nQuestion: {question}\n\nAnswer:"
    )
    print("Ask a question from the document:- \n")
    while True:
        question = input("User:").strip()
        if question.lower() == 'exit':
            break
        output = qa_chain.invoke({"question": question})
        print(f"Answer: {output['answer']} \n")
    return qa_chain
if __name__=='__main__':
    d=pdf_to_doc(r"C:\Users\deban\OneDrive\Desktop\(McGraw-Hill Series in Mechanical Engineering) Jack Holman - Heat Transfer-McGraw-Hill Science_Engineering_Math (2009).pdf")
    vectorstore=vectorize(d)
    qa_agent=create_qa_agent(vectorstore)