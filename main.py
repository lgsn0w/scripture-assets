import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

# Set up logging to see errors in Vercel Dashboard
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app) # Allow all connections

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

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
        # FORCE JSON parsing (fixes 400 errors if headers are missing)
        data = request.get_json(force=True, silent=True)
        
        if not data:
            logger.error("Failed to parse JSON")
            return jsonify({"answer": "Error: I could not understand the message format."}), 400

        user_messages = data.get('messages', [])
        logger.info(f"Received {len(user_messages)} messages")

        if not user_messages:
            return jsonify({"answer": "I am here. What is on your heart?"})

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}] + user_messages

        chat_completion = client.chat.completions.create(
            messages=conversation,
            model="llama3-8b-8192",
            temperature=0.7,
        )

        answer = chat_completion.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Server Error: {e}")
        return jsonify({"answer": "I apologize, but I am having trouble connecting right now."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)