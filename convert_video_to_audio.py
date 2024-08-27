import ffmpeg
import sys

def convert_video_to_audio(video_path, output_audio_path):
    try:
        print(f"Converting video to audio...")

        # Use ffmpeg to extract audio and save it as a .wav file
        ffmpeg.input(video_path).output(output_audio_path).run()

        print(f"Audio file saved at: {output_audio_path}")
    except Exception as e:
        print(f"Error converting video to audio: {str(e)}")
        raise e

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_video_to_audio.py <input_video_path> <output_audio_path>")
        sys.exit(1)

    video_file = sys.argv[1]
    output_audio_file = sys.argv[2]

    convert_video_to_audio(video_file, output_audio_file)