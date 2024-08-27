import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import sys

def preprocess_audio(input_file_path: str, output_file_path: str, target_sr=16000):
    """
    Preprocess the audio by resampling, noise reduction, trimming silence, and normalizing.
    The processed audio is saved to the output file path.

    Parameters:
    - input_file_path (str): Path to the input audio file.
    - output_file_path (str): Path to save the processed audio file.
    - target_sr (int): Target sampling rate for resampling (default: 16000).
    """
    # Load audio file
    print(f"Loading audio file: {input_file_path}")
    y, sr = librosa.load(input_file_path, sr=None)
    
    # Resample audio to the target sampling rate
    print(f"Resampling audio to {target_sr} Hz")
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    # Reduce noise in the audio
    print("Reducing noise in the audio")
    y_denoised = nr.reduce_noise(y=y_resampled, sr=target_sr)
    
    # Trim leading and trailing silence from the audio
    print("Trimming silence from the audio")
    y_trimmed, _ = librosa.effects.trim(y_denoised)
    
    # Normalize the audio to have a maximum value of 1.0
    print("Normalizing the audio")
    y_normalized = y_trimmed / np.max(np.abs(y_trimmed))
    
    # Save the processed audio to the specified output path
    print(f"Saving processed audio to: {output_file_path}")
    sf.write(output_file_path, y_normalized, target_sr)
    
    print("Audio preprocessing complete")
    return output_file_path

if __name__ == "__main__":
    # Check if the correct number of arguments are passed
    if len(sys.argv) != 3:
        print("Usage: python preprocess_audio.py <input_audio_file> <output_audio_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Call the preprocessing function
    preprocess_audio(input_file_path, output_file_path)
