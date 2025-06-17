# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "semantic-text-splitter",
#     "numpy",
#     "tqdm",
#     "google-genai",
#     "httpx",
#     "pillow",  
#     "argparse",
#     "fastapi", 
#     "html2text", 
#     "html2text", 
#     "markdownify",
#     "uvicorn",
#  ]
# ///

import argparse
import base64
import json
import numpy as np
import os
import re
import time
from pathlib import Path
from fastapi import FastAPI,Request
from pydantic import BaseModel
import httpx
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    image: str = None

              
class RateLimiter:
    def __init__(self,requests_per_minute=4, requests_per_second=2):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0


    def wait_if_needed(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < (1.0 / self.requests_per_second):
            sleep_time = (1.0 / self.requests_per_second)- time_since_last
            time.sleep(sleep_time)

        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]

        self.request_times.append(current_time)
        self.last_request_time = current_time 

rate_limiter = RateLimiter(requests_per_minute=5, requests_per_second=2) 

def get_image_description(image_path):
    """Get a description of the image using Google GenAI."""
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    my_file = client.files.upload(file=image_path)

    # with open(os.path.join('pdsaiitm.github.io', image_path), 'rb') as image_file:
    #     image_data = image_file.read()
        

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, "Describe the content of this image in detail,focusing on any text, objects,or relevant features that could help answer questions about it."],
    )
    
    #print(response)
    return response.text

def get_embedding(text: str, max_retries: int =10) -> list[float]:
    """Get embedding for the given text using Google GenAI."""
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    for attempt in range(max_retries):
        try:
            # Apply rate limiting
            rate_limiter.wait_if_needed()

            result = client.models.embed_content(
                model='gemini-embedding-exp-03-07',
                contents=[text]
            )
            
            embedding = result.embeddings[0].values
            return embedding
        
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded, waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            elif attempt == max_retries - 1:
                print(f"Failed to get embedding after {max_retries} attempts: {e}")
                raise
            else:
                print(f"Attempt {attempt + 1} failed: {e}, retrying..")
                time.sleep(1)
    raise Exception("Max retries exceeded")

def load_embeddings():
    """Load chunks and embeddings from npz file"""
    data = np.load("embeddings.npz", allow_pickle=True)
    return data["chunks"],data["embeddings"]
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            f"Answer the question based on the context provided.\n\nQuestion: {question}\n\nContext:{context}",
        ],
        config=GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            stop_sequences=["\n"]
        )#,
        #http_options=HttpOptions(timeout_seconds=60)
    )

def generate_llm_response(question:str, context: str):
    """Generate a response from the LLM using the question and context"""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    #use system prompr to instruct the model to answer based on the context
    system_prompt = "You are a helpful assistant. Use the provided context to answer the question."
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            system_prompt,
            f"Context: {context}",
            f"Question: {question}"
        ],
        config=GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.5,
            top_p=0.95,
            top_k=40
        )#,
        #http_options=HttpOptions(timeout_seconds=30)
    )
    return response.text


def answer(question: str, image: str = None):
    api_key = os.getenv("GOOGLE_API_KEY")
    loaded_chunks, loaded_embeddings = load_embeddings()
    if image:
        image_description = get_image_description(f"data:image/jpeg;base64,{image}")
        question += f"{image_description}"
    question_embedding = get_embedding(question)
    # Calculate cosine similarity
    similarities = np.dot(loaded_embeddings, question_embedding) / (
    np.linalg.norm(loaded_embeddings, axis=1) * np.linalg.norm(question_embedding)
    )

    # Get the index of the 10 most similar chunks
    top_indices = np.argsort(similarities)[-10:][::-1]
    top_chunks = [loaded_chunks[i] for i in top_indices]

    
    response = generate_llm_response(question,"\n".join(top_chunks))
    return {
        "question": question,
        "response": response,
        "top_chunks": top_chunks
    }


@app.post("/api")
async def api_answer(request:Request):
    try :
        data =  await request.json()
        #print(data)
        response = answer(data.get("question"),data.get("image"))
        answer_ = response['response']
        links = []
        return {
            "answer": answer_
            "links": links,
        }
    except Exception as e:
        return {"error": str(e)}
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host= "0.0.0.0",port=8222)

