from flask import Flask, request, jsonify, render_template
from model import ask_ai  

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({'response': 'Please send a valid message.'}), 400

    response = ask_ai(message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
