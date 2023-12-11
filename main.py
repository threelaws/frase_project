
from typing import Optional
from fastapi import FastAPI,Form
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)
API_key = os.environ['OPENAI_API_KEY']

app = FastAPI()

origins = [
    "http://localhost:3000",  # Allow CORS requests from this origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {'hello': 'message'}

from pydantic import BaseModel
class Bot(BaseModel):#
   question:str
   site:str
@app.post("/chatbot/")
async def submit(question: str = Form(...), site: str = Form(...)):

    
    try:
        if site:
            print(site)
            website = site
        else:
            website = 'https://brain.d.foundation/Blockchain/Foundational+topics/Blocks'

        if question:
           print(question)
           
           if question.__contains__('?'):
               question = question.replace('?', '')
        else:
            question = 'what is thos site about?'
        website = website.replace('https://', '')
        website = website.replace('http://', '')
        
        url = 'https://' + website
        print(url)
        data = get_scrape_data_from_url(url)
        print (f'You have {len(data)} pages in your data')
        chunks = chunk_data(data)
        print(len(chunks))
        vector_store = create_embeddings(chunks)
        if len(data) < 3:
            k = 1
        else:
            k = 3

        answer = ask_and_get_answer(vector_store, question, k)
    

    #return {"question": chatbot.q, "site": chatbot.website, "answer": answer}
        return {"question": question, "answer": answer}
            
    except Exception as e:
        return f"An error occurred in QandA: {str(e)}"
    

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=25) 
    chunks = text_splitter.split_documents(data) 
    return chunks

    
# print_embedding_cost(chunks)

def create_embeddings(chunks):
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        return f"An error occurred in create_embeddings: {str(e)}"

def ask_and_get_answer(vector_store, q, k):
    try:
        from langchain.chains import RetrievalQA
        from langchain.chat_models import ChatOpenAI

        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)
        retriever = vector_store.as_retriever(search_type='similarity_score_threshold' , search_kwargs={'k': k, 'score_threshold': 0.5})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        
        answer = chain.run(q)
        return answer
    except Exception as e:
        return f"An error occurred in ask_and_get_answer: {str(e)}"

def gettext_website(website):
    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(website)
        soup = BeautifulSoup(response.text, 'html.parser')

        text = ' '.join([p.text for p in soup.find_all('p')])
        if text == '':
            text = ' '.join([p.text for p in soup.find_all('span')])

        
        return text
    except Exception as e:
        return f"An error occurred in gettext_website: {str(e)}"

def get_scrape_data_from_url(your_url):
    try:
        from langchain.document_loaders import WebBaseLoader

        loader = WebBaseLoader(your_url)
        scrape_data = loader.load()
        print(f'You have {len(scrape_data)} pages in your data')
        return scrape_data
    except Exception as e:
        return f"An error occurred in get_scrape_data_from_url: {str(e)}"


    