import openai
import re

from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI API client
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a function to extract surrounding text for the query
def extract_surrounding_text(document, query, window_size=50):
    # Find the query in the document
    match = re.search(rf'\b{re.escape(query)}\b', document)
    if match:
        start = max(match.start() - window_size, 0)
        end = min(match.end() + window_size, len(document))
        surrounding_text = document[start:end]
        return surrounding_text
    else:
        return None

# Define the query and the document text
query = "bicycle"
document = """
The invention of the bicycle revolutionized personal transport. Early bicycles had large wheels and pedals. 
They were initially considered a luxury item, but soon became widely popular. The history of the bicycle is 
marked by continuous innovation and improvements, making it an enduring mode of transportation.
"""

# Extract the surrounding text for the query
surrounding_text = extract_surrounding_text(document, query)

if surrounding_text:
    # Combine query with surrounding text to form context
    context = f"The context for the query '{query}' is: {surrounding_text}"
    print(context)
    # Generate context embedding using OpenAI
    response = openai.Embedding.create(
        input=context,
        model="text-embedding-ada-002"
    )

    # Extract the embedding vector
    embedding = response['data'][0]['embedding']

    # Print the embedding vector
    print(embedding)
else:
    print(f"Query '{query}' not found in the document.")
