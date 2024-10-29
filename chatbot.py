import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and knowledge base
embeddings = np.load('embeddings.npy')
with open('documents.json', 'r') as f:
    knowledge_base = json.load(f)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_most_relevant_doc(query):
    # Encode the user's query
    query_embedding = model.encode([query])

    # Compute cosine similarities between the query and document embeddings
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Find the index of the most similar document
    best_match_idx = np.argmax(similarities)
    return knowledge_base[best_match_idx]

def format_sentence(text):
    """Ensure the text is properly capitalized and ends with punctuation."""
    # Remove any extra whitespace
    text = text.strip()
    
    # Ensure the first letter is uppercase, even if the text contains leading spaces
    if len(text) > 0:
        first_char = text[0].upper()
        rest = text[1:]

        # Combine the corrected first character with the rest of the sentence
        text = first_char + rest

    # Ensure proper punctuation at the end of the sentence
    if not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

def chatbot():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        query = input("You: ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Retrieve the most relevant document
        relevant_doc = get_most_relevant_doc(query)
        response = relevant_doc['content']
        
        # Format the response to ensure proper sentence case
        response = format_sentence(response)

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
