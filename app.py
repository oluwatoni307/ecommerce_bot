from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from bot import bot  # Import the bot function from your existing code

app = Flask(__name__)
CORS(app)  # This allows CORS for all domains on all routes


def extract_latest_question(conversations):
    # Get only the last 10 conversations (both user questions and bot answers)
    last_conversations = conversations[-10:]

    # Assuming that each conversation has a structure like {'user': 'user question', 'bot': 'bot answer'}
    # Extract the latest user question
    if last_conversations and "user" in last_conversations[-1]:
        latest_question = last_conversations[-1]["user"]
    else:
        latest_question = ""

    return last_conversations, latest_question


import json
import time


def generate_response(context, user_question):
    # This function should yield partial responses
    # You'll need to modify your bot function to yield partial results
    try:
        for partial_response in bot(context=context, user_question=user_question):
            yield f"data: {json.dumps({'response': partial_response})}\n\n"
            time.sleep(0.1)  # Small delay to control the stream rate
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    conversations = data.get("conversations", "")
    conversation, userquery = extract_latest_question(conversations)

    return Response(
        generate_response(conversation, userquery), content_type="text/event-stream"
    )


if __name__ == "__main__":
    app.run(debug=True)
