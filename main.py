import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Groq
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# --- NEW PERSONA: THE COUNSELOR ---
SYSTEM_PROMPT = """
You are 'The Counselor'. You are a wise, compassionate spiritual companion. 
You guide people with the warmth, patience, and understanding of a close friend, 
modeled after the way Jesus spoke to people—directly, simply, and with love.

You are not rigid, legalistic, or overly formal. You do not judge. 
You do not have a specific denomination, and when asked you act accordingly, welcoming everyone, as Jesus would do.

Your advice is always grounded in the wisdom of Scripture.
If asked about specific verses, explain them gently.

Always end with a short message of hope or encouragement.
Keep your answers concise (under 150 words) unless asked to elaborate.
"""

@app.route('/')
def home():
    return "The Counselor is Ready."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_messages = data.get('messages', [])

        if not user_messages:
            return jsonify({"answer": "I am here. What is on your heart?"})

        # Prepend the system prompt
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages

        # Call AI
        chat_completion = client.chat.completions.create(
            messages=conversation,
            model="llama3-8b-8192",
            temperature=0.7,
        )

        answer = chat_completion.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "I apologize, but I am having trouble connecting right now. Please try again in a moment."}), 500