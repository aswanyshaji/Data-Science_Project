import os
import openai
import re
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "./chroma_db"
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

def convert_date_format(date_str: str) -> str:
    """Convert date from DD-MM-YYYY to YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None

def perform_similarity_search_by_metadata(date: str):
    try:
        # Convert the date to YYYY-MM-DD format
        formatted_date = convert_date_format(date)
        if not formatted_date:
            return []

        # Perform the search based on metadata
        print(f"Performing search for date: {formatted_date}")
        matching_docs = chroma_db.similarity_search_with_score(
            query="",
            k=100,
            filter={"meeting_date": formatted_date}
        )
        print(f"Number of documents found: {len(matching_docs)}")
        for doc, score in matching_docs:
            print(f"Document metadata: {doc.metadata}, score: {score}")
        return [doc for doc, score in matching_docs if doc.metadata.get("meeting_date") == formatted_date]
    except Exception as e:
        print(f"Error during similarity search: {str(e)}")
        return []

def perform_general_similarity_search(query: str):
    try:
        # Perform a general search across all documents
        print(f"Performing general search for query: {query}")
        matching_docs = chroma_db.similarity_search_with_score(query=query, k=5)
        print(f"Number of documents found: {len(matching_docs)}")
        for doc, score in matching_docs:
            print(f"Document metadata: {doc.metadata}, score: {score}")
        return [doc for doc, score in matching_docs]
    except Exception as e:
        print(f"Error during general similarity search: {str(e)}")
        return []

def generate_response_from_gpt4(query: str, context: str):
    try:
        prompt = f"""
You are the Meeting Minute Bot. Your task is to provide accurate information based on the provided meeting minutes. 
Respond only with relevant information found in the context. If the information is not available, indicate that you cannot provide it.

Context: {context}

User's Query: {query}

Please provide an answer strictly based on the above context. if user asks anything out of scope you need to anser like Your question seems outside the stored meeting minutes. 
I'm happy to help with anything within them—just ask! 
"""

        response = openai.ChatCompletion.create(
            model = "gpt-4",
            messages = [{"role": "system", "content": prompt}],
            temperature = 0  # Set temperature to 0 for more deterministic and focused responses
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating response from GPT-4: {str(e)}")
        return "I'm sorry, I encountered an error while trying to generate the response."
def get_answer(query: str):
    # Normalize the input to lowercase and strip any leading/trailing whitespace
    normalized_query = query.lower().strip()

    # Define a set of greeting words
    greetings = {"hi", "hello", "hey", "hai", "greetings", "hola", "howdy"}

    # Check if the input is a greeting
    if any(word in normalized_query for word in greetings):
        return ("Hello! I am the Meeting Minute Bot, always ready to assist you with "
                "any information you need from your meeting minutes. Whether it's key points, "
                "summaries, or full transcripts, I'm here to help!")

    # Proceed with normal query processing if it's not a greeting
    date = extract_date_from_query(query)

    query_lower = normalized_query
    section = None
    if "entire minute" in query_lower or "full minute" in query_lower:
        if not date:
            return "Please provide a specific date in the format DD-MM-YYYY to retrieve the entire meeting minutes."
        section = "full_document"
    elif "abstract summary" in query_lower:
        if not date:
            return "Please provide a specific date in the format DD-MM-YYYY to retrieve the abstract summary."
        section = "abstract_summary"
    elif "key points" in query_lower:
        if not date:
            return "Please provide a specific date in the format DD-MM-YYYY to retrieve the key points."
        section = "key_points"

    if section:
        matching_docs = perform_similarity_search_by_metadata(date)
        if len(matching_docs) == 0:
            return f"No meeting minutes found for {date}."
        context = "\n".join([doc.page_content for doc in matching_docs])
    else:
        matching_docs = perform_general_similarity_search(query)
        if len(matching_docs) == 0:
            return "I couldn't find relevant information in the stored meeting minutes."
        context = " ".join([doc.page_content for doc in matching_docs])

    answer = generate_response_from_gpt4(query, context)
    return answer

def extract_date_from_query(query):
    # Basic function to extract a date from the query (assuming format DD-MM-YYYY)
    match = re.search(r'\d{2}-\d{2}-\d{4}', query)
    return match.group(0) if match else None

# Example usage
if __name__ == "__main__":
    # Display the welcome message
    print("Hi, I am the Meeting Minute Bot. I can help you retrieve information from any past meeting.")
    print("You can ask me about key points, action items, summaries, or even the full minutes from specific meetings.")

    # Prompt the user to enter their query
    query = input("\nPlease enter your query: ")
    print(get_answer(query))