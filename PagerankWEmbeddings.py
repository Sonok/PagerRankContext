#pip install requests beautifulsoup4 openai scikit-learn python-dotenv
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import time


# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI API client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to get the text content of a webpage
def get_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extract text from paragraphs
    paragraphs = soup.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs])
    return content

# Function to split text into smaller chunks
def split_into_chunks(text, max_tokens=8000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1  # Add 1 for the space
        if current_length > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = len(word) + 1
        current_chunk.append(word)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Function to generate embeddings for a given text using OpenAI with retries
def generate_embedding(text, retries=3, backoff_factor=1.0):
    chunks = split_into_chunks(text)
    embeddings = []
    for chunk in chunks:
        for attempt in range(retries):
            try:
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[chunk]
                )
                embeddings.append(response['data'][0]['embedding'])
                break  # Break out of the retry loop if successful
            except openai.error.APIError as e:
                if attempt < retries - 1:  # If not the last attempt, wait before retrying
                    wait_time = backoff_factor * (2 ** attempt)
                    print(f"API error occurred: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise  # Raise the exception if the last attempt fails
    # Average the embeddings of all chunks to get a single embedding for the text
    avg_embedding = np.mean(embeddings, axis=0)
    return avg_embedding

# Function to calculate cosine similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

# List of URLs to crawl
urls = [
    'https://www.apple.com/',
    'https://en.wikipedia.org/wiki/Apple',
    'https://en.wikipedia.org/wiki/Apple_Inc.'
]

# Dictionary to store embeddings
embeddings = {}

# Crawl each URL and generate embeddings
for url in urls:
    content = get_page_content(url)
    embedding = generate_embedding(content)
    embeddings[url] = embedding

# Calculate similarity between each pair of pages
similarity_matrix = np.zeros((len(urls), len(urls)))

for i, url1 in enumerate(urls):
    for j in range(i, len(urls)):
        if i != j:
            similarity = calculate_similarity(embeddings[url1], embeddings[urls[j]])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
        else:
            similarity_matrix[i][j] = 1

# Display similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)

