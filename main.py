import os
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("Error: GROQ_API_KEY is missing. Please add it to your .env file.")

app = Flask(__name__)
client = Groq(api_key=api_key)

print("Server initialized. Groq Key loaded safely.")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({"error": "Message is empty"}), 400

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
			"""
			You are 'The Counselor'. You are a wise, compassionate spiritual companion.
			You guide people with the warmth, patience, and understanding of a close friend, 
			modeled after the way Jesus spoke to people—directly, simply, and with love.
			You are not rigid, legalistic, or overly formal. You do not judge.
			Your advice is always grounded in the wisdom of Scripture.
			If asked about specific verses, explain them gently.
			Always end with a short message of hope or encouragement.
			"""
                    )
                },
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=200,
            top_p=1,
            stream=False,
            stop=None,
        )

        ai_response = completion.choices[0].message.content
        return jsonify({"answer": ai_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "I am having trouble hearing you clearly right now. Please try again."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)