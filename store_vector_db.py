import uuid
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import sys

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_function = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize ChromaDB client and create a collection
persist_directory = "./chroma_db"
os.makedirs(persist_directory, exist_ok=True)
collection = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def store_vector_db(minutes_path, document_name, meeting_date):
    try:
        with open(minutes_path, "r") as f:
            minutes_text = f.read()

        # Generate a unique document ID
        document_id = str(uuid.uuid4())

        # Create metadata for the document
        metadata = {
            "document_id": document_id,
            "document_name": document_name,
            "meeting_date": meeting_date
        }

        # Create embeddings for the entire document
        embeddings = model.encode([minutes_text])

        # Store in ChromaDB
        collection.add_texts(
            texts=[minutes_text],
            embeddings=[embedding.tolist() for embedding in embeddings],
            metadatas=[metadata],
        )

        print("Data successfully stored in ChromaDB.")

    except Exception as e:
        print(f"Error storing minutes in ChromaDB: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python store_vector_db.py <minutes_path> <document_name> <meeting_date>")
    else:
        minutes_path = sys.argv[1]
        document_name = sys.argv[2]
        meeting_date = sys.argv[3]
        store_vector_db(minutes_path, document_name, meeting_date)