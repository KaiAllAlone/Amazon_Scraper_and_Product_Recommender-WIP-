from flask import Flask, render_template, request, jsonify
from recommender_engine import setup_engine, ask_bot
import asyncio

app = Flask(__name__)

# Run async setup at startupk
loop = asyncio.get_event_loop()
loop.run_until_complete(setup_engine())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    user_input = data.get('message', '').strip()
    print(f"[DEBUG] Received message: {user_input}")
    if not user_input:
        return jsonify({"reply": "Please enter a message!"})

    # Run async ask_bot function and serialize result
    reply = loop.run_until_complete(ask_bot(user_input))
    print(f"[DEBUG] Bot reply: {reply}")
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
