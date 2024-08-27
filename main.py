from fastapi import FastAPI, Form, UploadFile, File, Request, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import shutil
import subprocess
import mimetypes
import uuid
import time
import pandas as pd
from datetime import datetime
from typing import Dict
import sqlite3

from transcript_from_audio import generate_transcript
from generate_minute import generate_minutes
from minute_search import get_answer  # Import the RAG search functionality

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store file paths and meeting dates
session_data = {}

# SQLite database setup
DATABASE_FILE = 'qa_database.db'

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            response_time REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
# Initialize the database
init_db()


def is_video_file(file_path):
    mimetype, _ = mimetypes.guess_type(file_path)
    return mimetype and mimetype.startswith('video')

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), meeting_date: str = Form(...)):
    try:
        directory = f"data/{meeting_date}"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        session_data['uploaded_file_path'] = file_path
        session_data['meeting_date'] = meeting_date
        print(f"File uploaded to: {file_path}")
    
        return {"status": "success", "message": "File uploaded successfully."}
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        return {"status": "error", "message": f"File upload failed: {str(e)}"}

@app.post("/generate_minutes/")
async def generate_minutes_endpoint():
    try:
        uploaded_file_path = session_data.get('uploaded_file_path')
        meeting_date = session_data.get('meeting_date')

        if not uploaded_file_path or not meeting_date:
            return {"status": "error", "message": "No file uploaded or meeting date provided."}

        # Check if the uploaded file is a video and convert it to audio if necessary
        if is_video_file(uploaded_file_path):
            audio_file_path = os.path.splitext(uploaded_file_path)[0] + ".wav"
            subprocess.run(
                ["python", "convert_video_to_audio.py", uploaded_file_path, audio_file_path],
                check=True
            )
        else:
            audio_file_path = uploaded_file_path

        # Define the path for the preprocessed audio file
        preprocessed_file_path = f"data/{meeting_date}/preprocessed_{os.path.basename(audio_file_path)}"

        # Ensure the preprocessed file path has the correct .wav extension
        if not preprocessed_file_path.lower().endswith(".wav"):
            preprocessed_file_path += ".wav"

        # Call the external preprocessing script
        subprocess.run(
            ["python", "preprocess_audio.py", audio_file_path, preprocessed_file_path],
            check=True
        )

        # Now generate the transcript using the preprocessed audio file
        print(f"Generating transcript for: {preprocessed_file_path}")
        transcript_path = generate_transcript(preprocessed_file_path)
        print(f"Transcript generated at: {transcript_path}")

        # Generate minutes and save as PDF
        minutes_path, pdf_path = generate_minutes(transcript_path, meeting_date)
        print(f"Minutes generated at: {minutes_path}")
        print(f"PDF generated at: {pdf_path}")

        # Store the generated minutes in the vector database
        document_name = f"Minutes for {meeting_date}"
        subprocess.run(["python", "store_vector_db.py", minutes_path, document_name, meeting_date])

        return {
            "status": "success", 
            "pdf_url": f"http://127.0.0.1:8000/download/{meeting_date}/{os.path.basename(pdf_path)}"
        }
    except subprocess.CalledProcessError as e:
        print(f"Processing failed: {str(e)}")
        return {"status": "error", "message": f"Processing failed: {str(e)}"}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/download/{meeting_date}/{filename}")
async def download_file(meeting_date: str, filename: str):
    file_path = os.path.join("data", meeting_date, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/pdf', filename=filename)
    else:
        return {"status": "error", "message": "File not found."}

@app.post("/search/")
async def search_endpoint(request: Request, response: Response, query: str = Form(...)):
    try:
        print(f"Received query: {query}")
        
        # Start timing for database retrieval
        db_start_time = time.time()
        
        # Check if the answer is already in the SQLite database
        answer, response_time = get_answer_from_db(query)
        
        if answer:
            # Calculate how long it took to retrieve the answer from the database
            db_end_time = time.time()
            response_time = db_end_time - db_start_time
            print(f"Answer retrieved from database in {response_time:.4f} seconds")
        else:
            # Start timing the response generation
            start_time = time.time()

            # Call the search function or AI model to get the answer
            answer = get_answer(query)

            # End timing the response generation
            end_time = time.time()
            response_time = end_time - start_time
            print(f"Generated answer in {response_time:.4f} seconds")

            # Store the new query, answer, and response time in the SQLite database
            store_answer_in_db(query, answer, response_time)

        # Print the question, answer, and response time to the console
        print(f"Question: {query}")
        print(f"Answer: {answer}")
        print(f"Response Time: {response_time:.4f} seconds")

        return {
            "status": "success", 
            "query": query, 
            "answer": answer, 
            "response_time": f"{response_time:.4f} seconds"
        }
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_answer_from_db(query: str):
    """Retrieve answer from the SQLite database if it exists."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT answer, response_time FROM qa_data WHERE query = ?", (query,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        answer, response_time = result
        return answer, response_time
    return None, None

def store_answer_in_db(query: str, answer: str, response_time: float):
    """Store the new query, answer, and response time in the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO qa_data (query, answer, response_time, timestamp) VALUES (?, ?, ?, ?)",
        (query, answer, response_time, timestamp)
    )
    conn.commit()
    conn.close()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

