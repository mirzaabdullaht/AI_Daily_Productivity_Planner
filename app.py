import os
import gradio as gr
import whisper
import datetime
import re
from groq import Groq

# Load the Whisper model
model = whisper.load_model("base")

# Load the Groq API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def extract_dates(text):
    # Refined pattern to extract time ranges (e.g., 9:00 am - 10:00 am)
    pattern = r'(\d{1,2}:\d{2} [ap]m) - (\d{1,2}:\d{2} [ap]m)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        start_time_str, end_time_str = match.groups()
        now = datetime.datetime.utcnow()
        try:
            start_time = datetime.datetime.strptime(start_time_str, "%I:%M %p").replace(year=now.year, month=now.month, day=now.day)
            end_time = datetime.datetime.strptime(end_time_str, "%I:%M %p").replace(year=now.year, month=now.month, day=now.day)
            return start_time, end_time
        except ValueError:
            return None, None
    return None, None

def process_input(audio_file, text_input):
    try:
        transcription_text = ""
        if audio_file:
            # Process audio file
            audio = whisper.load_audio(audio_file)
            result = model.transcribe(audio)
            text = result['text']
            transcription_text = text  # Store transcription for display
        elif text_input:
            # Use text input directly
            text = text_input
        else:
            return "No input provided", None, None

        # Refine the prompt for concise responses
        refined_prompt = (
            f"Please create a concise schedule for the following input: '{text}'. "
            f"Break down the schedule into 30-minute intervals, specifying start and end times for each task or activity. "
            f"Ensure the plan covers the entire day from start to end, and provide clear transitions between activities."
        )

        # Generate a response using Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": refined_prompt}],
            model="llama3-8b-8192",  # Adjust if necessary
        )
        response_message = chat_completion.choices[0].message.content.strip()

        # Print the response message for debugging
        print(f"Response Message: {response_message}")

        # Extract start and end times from the response
        start_time, end_time = extract_dates(response_message)

        if start_time and end_time:
            summary = "Scheduled Task"
            description = response_message
            return transcription_text, response_message, f"Start: {start_time}, End: {end_time}"
        else:
            return transcription_text, "Could not extract start and end times from the response", None

    except Exception as e:
        return f"An error occurred: {e}", None, None

iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio File (optional)"),
        gr.Textbox(placeholder="Or type your text here...", label="Text Input (optional)")
    ],
    outputs=[
        gr.Textbox(label="Transcription Text"),  # Show transcription text here
        gr.Textbox(label="Response Text"),
        gr.Textbox(label="Extracted Time Range")
    ],
    live=True,
    title="AI Daily Productivity Planner",
    description="Upload an audio file or type text input. The AI planner will transcribe or process it, and extract start and end times."
)

iface.launch()
