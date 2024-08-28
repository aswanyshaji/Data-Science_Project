# AI-Meeting Minute Assistant - AI-Powered Automated Meeting Minute Generation and Intelligent Search System

## Overview
In today's fast-paced business environment, effective communication and efficient information management are critical for organizational success. Meetings are central to decision-making processes, project planning, and overall organizational efficiency. However, manually taking and organizing meeting minutes can be time-consuming and prone to errors. To address these challenges, this project automate the generation of meeting minutes from audio or video recordings. It also features an intelligent search system that allows users to retrieve specific information from stored meeting minutes using a chatbot interface.
## Application
This application is designed to be used by organizations and businesses to automate their meeting documentation process. The application is particularly useful in scenarios where accurate, timely, and well-organized meeting records are essential, such as in corporate environments, governmental agencies, and academic institutions.
## Requirements
- The project was initiated by creating a virtual environment using Anaconda to manage dependencies efficiently.  fter setting up the environment, all required packages were installed using the requirements.txt file. This setup ensures an isolated environment for the project, managing dependencies effectively.
## Development Environment and code Structure
The project was developed using the VS Code editor, with the front end built in HTML and the back end implemented using FastAPI. The main program is centered around FastAPI, which serves as the core of the application's backend logic. The project is organized in such a way that all functionalities are called through FastAPI. The FastAPI logic is distinctly separated from other components. This architecture ensures that the main FastAPI program efficiently handles API requests and responses, while other functionalities are invoked as needed. The frontend, designed using HTML, provides a user-friendly interface that interacts with the FastAPI backend. This structured approach enhances the maintainability and clarity of the project, facilitating easy extension and modification.
## Workflow of the Application

**Overview**: The application is built around FastAPI, which serves as the core backend, managing API requests and responses. The system automatically processes user-uploaded audio or video files through a sequence of scripts, each handling a specific task. Below is the step-by-step workflow of the application:

1. **User Uploads File**:
   - The user uploads an audio (`.wav`) or video (`.mp4`) file through the frontend interface (`front_interface.html`).

2. **Convert Video to Audio** (if applicable):
   - If the uploaded file is a video (`.mp4`), the system invokes `convert_video_to_audio.py`.
   - **Input**: Video file (`.mp4`).
   - **Output**: Audio file (`.wav`).

3. **Preprocess Audio**:
   - The output audio file (`.wav`) is then processed by `preprocess_audio.py`.
   - **Purpose**: Clean and prepare the audio for transcription.

4. **Transcription from Audio**:
   - The cleaned audio is transcribed using `transcript_from_audio.py`, which employs the Whisper model.
   - **Output**: Transcription text from the audio.

5. **Preprocess Transcript**:
   - The transcribed text is further processed by `preprocess_transcript.py`.
   - **Purpose**: To format or refine the transcription for minute generation.

6. **Generate Minutes**:
   - The refined transcript is used to generate meeting minutes via `generate_minute.py`.
   - **Output**: A document or text file containing the meeting minutes.
   - The user can download the generated minutes from the frontend interface.

7. **Store Vector Embedding in Database**:
   - The generated minutes are converted into vector embeddings.
   - `store_vector_db.py` is called to store these embeddings in a vector database (Chroma DB).
   - **Purpose**: For future search, retrieval, or analysis purposes.

**These seven processes are executed one after another automatically after the user uploads the audio or video file.**

### Retrieval of Specific Meeting Information

When the user wants to retrieve specific meeting information, they can search through the chatbot interface on the frontend. The following steps outline this process:

1. **User Query**:
   - The user enters a query in the chatbot interface on the frontend.

2. **Processing the Query**:
   - The FastAPI main program receives the query and calls `minute_search.py`.
   
3. **Search and Retrieval**:
   - `minute_search.py` processes the query and searches the stored vector embeddings in the Chroma DB.
   - **Output**: Retrieves and returns the relevant meeting information or answers corresponding to the user’s query.

4. **Display Results**:
   - The retrieved information is displayed back to the user through the chatbot interface, providing answers or relevant meeting data based on the query.

### Additional Standalone Programs

1. **Research on Text Summarisation** (`research_on_text_summaraisation.py`)

   This program conducts research on various text summarization techniques, comparing traditional NLP algorithms like LexRank and TextRank to advanced LLM models such as BART, T5, and GPT-4. The performance of these summarization techniques is evaluated using ROUGE and METEOR scores.

   - **Purpose**: To analyze and compare the effectiveness of different text summarization methods.
   - **Techniques Covered**: Traditional algorithms (LexRank, TextRank) and advanced models (BART, T5, GPT-4).
   - **Evaluation Metrics**: ROUGE and METEOR scores.

2. **Minute Fetching Analysis** (`minute_fetching_analysis.py`)

   This program analyzes the accuracy and performance of the minute retrieval system. It measures the accuracy of each answer using BERTScore and evaluates the optimization results by employing SQLite to store previously asked questions and answers. The response times are measured for both optimized and non-optimized cases.

   - **Purpose**: To assess the accuracy of the minute retrieval system and optimize performance.
