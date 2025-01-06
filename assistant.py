import openai
import tempfile
import playsound

openai.api_key = "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"

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import os



def speak(text):
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save the speech to a temporary file in memory and play it directly
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        tts.save(temp_file.name)
        playsound.playsound(temp_file.name)



# Set up the text-to-speech engine
engine = pyttsx3.init()

def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your command...")
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print("Sorry, I couldn't understand. Please try again.")
        return None

# Function to get response from ChatGPT using the new API method

def therapist_response(user_input):
    prompt = f"""
    You are a compassionate, empathetic therapist. Your role is to listen actively, provide understanding, and offer thoughtful insights that help individuals explore their feelings, emotions, and thoughts. You should aim to make the user feel heard, supported, and understood. When responding, maintain a gentle, non-judgmental, and professional tone. Your responses should be focused on helping individuals reflect on their emotions and challenges, offering coping strategies, and fostering a positive, healthy mindset.

    The user may share their feelings, personal experiences, or struggles. In response, ask open-ended questions that encourage further exploration, provide affirmations, and gently guide them toward solutions or self-reflection. Be patient and kind in all your responses.
    """

    # Using messages to structure the conversation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" depending on your access
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )

    return response['choices'][0]['message']['content']


# Main loop
def main():
    while True:
        user_input = listen_to_user()
        if user_input:
            chatgpt_response = therapist_response(user_input)
            print(f"ChatGPT says: {chatgpt_response}")
            speak(chatgpt_response)

if __name__ == "__main__":
    main()