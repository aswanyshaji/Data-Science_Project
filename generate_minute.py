import openai
import os
from fpdf import FPDF

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_minutes(transcript_path, meeting_date):
    try:
        with open(transcript_path, "r") as f:
            transcript = f.read()

        def process_transcription(transcription):
            """Process the entire transcription by dividing it into smaller sections."""
            sections = split_into_sections(transcription)
            results = {
                'abstract_summary': '',
                'key_points': [],
                'action_items': [],
                'sentiment': 'Neutral'  # Placeholder for sentiment
            }
            for section in sections:
                section_results = meeting_minutes(section)
                results['abstract_summary'] += section_results['abstract_summary'] + '\n\n'
                results['key_points'].extend(section_results['key_points'])
                results['action_items'].extend(section_results['action_items'])
            
            results['key_points'] = list(dict.fromkeys(results['key_points']))
            results['action_items'] = list(dict.fromkeys(results['action_items']))
            return results

        def split_into_sections(text, max_length=4096):
            words = text.split()
            sections = []
            current_section = []

            for word in words:
                if sum(len(w) for w in current_section) + len(word) < max_length:
                    current_section.append(word)
                else:
                    sections.append(" ".join(current_section))
                    current_section = [word]
            if current_section:
                sections.append(" ".join(current_section))
            return sections

        def meeting_minutes(transcription):
            abstract_summary = abstract_summary_extraction(transcription)
            key_points = key_points_extraction(transcription)
            action_items = action_item_extraction(transcription)
            return {
                'abstract_summary': abstract_summary,
                'key_points': key_points,
                'action_items': action_items
            }

        def abstract_summary_extraction(transcription):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0,
                messages=[
                    {"role": "system", "content": "Summarize this text into a concise abstract paragraph."},
                    {"role": "user", "content": transcription}
                ]
            )
            return response.choices[0].message.content

        def key_points_extraction(transcription):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an AI that identifies and lists the main points discussed in a text. Ensure each point is unique, important, and relevant. Do not repeat points across sections."},
                    {"role": "user", "content": transcription}
                ]
            )
            key_points = response.choices[0].message.content.split('\n')
            return [point.strip() for point in key_points if point.strip()]

        def action_item_extraction(transcription):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are an AI that extracts important and actionable items from a text. List only the unique and significant tasks or actions agreed upon in the discussion. If no important tasks are identified, leave it empty. Avoid repeating points across sections."},
                    {"role": "user", "content": transcription}
                ]
            )
            action_items = response.choices[0].message.content.split('\n')
            return [item.strip() for item in action_items if item.strip()]

        results = process_transcription(transcript)
        minutes_text = f"Abstract Summary:\n{results['abstract_summary']}\n\nKey Points:\n{'\n'.join(results['key_points'])}\n\nAction Items:\n{'\n'.join(results['action_items'])}"

        minutes_path = os.path.splitext(transcript_path)[0] + "_minutes.txt"
        with open(minutes_path, "w") as f:
            f.write(minutes_text)
        
        # Create PDF with meeting date as the name
        pdf_path = os.path.join(os.path.dirname(transcript_path), f"{meeting_date}_minutes.pdf")
        create_pdf(minutes_text, pdf_path, meeting_date)
        
        print("Minutes generation completed.")
        return minutes_path, pdf_path
    except Exception as e:
        print(f"Error in generate_minutes: {str(e)}")
        raise e

class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Meeting Date: {self.meeting_date}", 0, 0, "R")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", size=12)
        self.multi_cell(0, 10, body, align="J")
        self.ln()

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)

def create_pdf(minutes_text, pdf_path, meeting_date):
    pdf = PDF()
    pdf.meeting_date = meeting_date  # Pass the meeting date to the PDF class
    pdf.set_left_margin(10)
    pdf.set_right_margin(10)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_chapter("Meeting Minutes", minutes_text)
    pdf.output(pdf_path)
