import uuid
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import sys

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize ChromaDB client and create a collection
persist_directory = "./chroma_db"
os.makedirs(persist_directory, exist_ok=True)
collection = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def create_overlapping_chunks(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def extract_numbered_items(text):
    items = []
    for line in text.split("\n"):
        if line.strip() and line.strip()[0].isdigit():
            items.append(line.strip())
    return items

def store_embeddings_in_chromadb(collection, embeddings, text_chunks, metadata_list):
    collection.add_texts(
        texts=text_chunks,
        embeddings=[embedding.tolist() for embedding in embeddings],
        metadatas=metadata_list,
    )

def store_vector_db(minutes_path, document_name):
    try:
        with open(minutes_path, "r") as f:
            minutes_text = f.read()

        # Split the text into different sections
        general_summary = minutes_text.split("Key Points:")[0].split("Abstract Summary:")[1].strip()
        key_points = minutes_text.split("Key Points:")[1].split("Action Items:")[0].strip()
        action_items = minutes_text.split("Action Items:")[1].strip()

        # Create overlapping chunks for the general summary
        summary_chunks = create_overlapping_chunks(general_summary)

        # Extract key points and action items into individual chunks
        key_points_chunks = extract_numbered_items(key_points)
        action_items_chunks = extract_numbered_items(action_items)

        # Combine all chunks
        all_chunks = summary_chunks + key_points_chunks + action_items_chunks

        # Generate metadata for each chunk
        metadata_list = [{"document_id": document_name, "section": "general_summary", "chunk_id": i} for i in range(len(summary_chunks))] + \
                        [{"document_id": document_name, "section": "key_points", "chunk_id": i} for i in range(len(key_points_chunks))] + \
                        [{"document_id": document_name, "section": "action_items", "chunk_id": i} for i in range(len(action_items_chunks))]

        # Create embeddings for each chunk
        embeddings = model.encode(all_chunks)

        # Store in ChromaDB
        store_embeddings_in_chromadb(collection, embeddings, all_chunks, metadata_list)

        print("Minutes successfully uploaded to ChromaDB.")
    except Exception as e:
        print(f"Error storing minutes in ChromaDB: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python store_vector_db.py <minutes_path> <document_name>")
    else:
        minutes_path = sys.argv[1]
        document_name = sys.argv[2]
        store_vector_db(minutes_path, document_name)