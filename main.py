from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import shutil
import subprocess
from query_logic import get_answer
from transcript import generate_transcript
from generate_minute import generate_minutes

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

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), meeting_date: str = Form(...)):
    directory = f"data/{meeting_date}"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    session_data['uploaded_file_path'] = file_path
    session_data['meeting_date'] = meeting_date
    print(f"File uploaded to: {file_path}")

    return {"status": "success"}

@app.post("/generate_minutes/")
async def generate_minutes_endpoint():
    try:
        uploaded_file_path = session_data.get('uploaded_file_path')
        meeting_date = session_data.get('meeting_date')

        print(f"Generating transcript for: {uploaded_file_path}")
        transcript_path = generate_transcript(uploaded_file_path)
        print(f"Transcript generated at: {transcript_path}")

        minutes_path, pdf_path = generate_minutes(transcript_path, meeting_date)
        print(f"Minutes generated at: {minutes_path}")
        print(f"PDF generated at: {pdf_path}")

        document_name = f"Minutes for {meeting_date}"
        subprocess.run(["python", "store_vector_db.py", minutes_path, document_name])

        return {
            "status": "success", 
            "pdf_url": f"http://127.0.0.1:8000/download/{os.path.basename(os.path.dirname(uploaded_file_path))}/{os.path.basename(pdf_path)}"
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/download/{meeting_date}/{filename}")
async def download_file(meeting_date: str, filename: str):
    file_path = os.path.join("data", meeting_date, filename)
    return FileResponse(file_path, media_type='application/pdf', filename=filename)

@app.post("/query/")
async def query_minutes(request: Request, query: str = Form(...)):
    try:
        print("Received query:", query)
        answer = get_answer(query)
        print("Generated answer:", answer)
        return {"status": "success", "query": query, "answer": answer}
    except Exception as e:
        print("Error:", e)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)