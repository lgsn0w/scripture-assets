import os
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (Vercel will use the ones you added in the dashboard)
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Error: GROQ_API_KEY is missing. Please add it to your project settings.")

app = Flask(__name__)
client = Groq(api_key=api_key)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        # The Android app now sends a list of messages instead of a single string
        messages_history = data.get('messages', [])

        if not messages_history:
            return jsonify({"error": "No messages provided"}), 400

        # This is the 'brain' and personality of the AI
        system_msg = {
            "role": "system",
            "content": (
                """
                You are 'The Counselor'. You are a wise, compassionate spiritual companion. 
                You guide people with the warmth, patience, and understanding of a close friend, 
                modeled after the way Jesus spoke to people—directly, simply, and with love.
                You are not rigid, legalistic, or overly formal. You do not judge. You do not have a specific 				denomination, and when asked you act accordingly, welcoming everyone, as Jesus would do.
                Your advice is always grounded in the wisdom of Scripture.
                If asked about specific verses, explain them gently.
                Always end with a short message of hope or encouragement.
                """
            )
        }

        # Combine the fixed system persona with the conversation history sent by the app
        full_conversation = [system_msg] + messages_history

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=full_conversation,
            temperature=0.7,
            max_tokens=1500,             top_p=1,
            stream=False,
            stop=None,
        )

        ai_response = completion.choices[0].message.content
        return jsonify({"answer": ai_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "I am having trouble hearing you clearly right now. Please try again."}), 500

# This is for local testing; Vercel will ignore this and use its own entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)