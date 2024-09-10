from flask import Flask, request, jsonify
from flask_cors import CORS
from bot import bot  # Import the bot function from your existing code

app = Flask(__name__)
CORS(app)  # This allows CORS for all domains on all routes


def extract_latest_question(conversations):
    # Get only the last 10 conversations (both user questions and bot answers)
    last_conversations = conversations[-10:]
    
    # Assuming that each conversation has a structure like {'user': 'user question', 'bot': 'bot answer'}
    # Extract the latest user question
    if last_conversations and 'user' in last_conversations[-1]:
        latest_question = last_conversations[-1]['user']
    else:
        latest_question = ''
    
    return last_conversations, latest_question

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    conversation = data.get('conversation', '')
    
    
    last_conv, user_question = extract_latest_question(conversation)
    
    try:
        response = bot(context=last_conv, user_question=user_question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)