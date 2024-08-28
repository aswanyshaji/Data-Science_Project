import openai
from pydub import AudioSegment #Pydub is a library for manipulating audio files.
import os

#Set Open AI Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_transcript(audio_path):
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_path, format="wav")

        # Define chunk length (1 minute)
        chunk_length_ms = 1 * 60 * 1000  #Sets the chunk length to 1 minute (in milliseconds).
        # Split the audio into chunks
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        # Directory to save chunks
        os.makedirs("chunks", exist_ok=True)

        # Function to transcribe a chunk using OpenAI Whisper API
        def transcribe_chunk(chunk, chunk_index):
            chunk_path = f"chunks/chunk_{chunk_index}.wav"
            chunk.export(chunk_path, format="wav")
            with open(chunk_path, "rb") as audio_file:
                response = openai.Audio.transcribe(
                    file=audio_file,
                    model="whisper-1"
                )
                return response['text']

        # Transcribe each chunk and combine the results
        final_transcription = ""
        for i, chunk in enumerate(chunks):
            # Check if chunk size is within the limit
            chunk_path = f"chunks/chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")
            if os.path.getsize(chunk_path) > 25 * 1024 * 1024:
                print(f"Chunk {i+1} exceeds the size limit and will be skipped.")
                continue

            print(f"Transcribing chunk {i+1}/{len(chunks)}...")
            chunk_transcription = transcribe_chunk(chunk, i)
            final_transcription += chunk_transcription + "\n"
        
        # Save the final transcription to a file
        transcript_path = os.path.splitext(audio_path)[0] + "_transcript.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(final_transcription)
        
        print("Transcription completed.")
        return transcript_path
    except Exception as e:
        print(f"Error in generate_transcript: {str(e)}")
        raise e