from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from processor import handle_documents_and_qa , answer_question 
import shutil

from pymilvus import connections, Collection

def clear_vector_db():
    connections.connect("default", host="localhost", port="5000")
    collection = Collection("LangChainCollection")
    if not collection.is_empty:
        print("Clearing the vector database...")
    else:
        print("The vector database is already empty.")
        
    collection.load()
    collection.delete("true")  # deletes all data


UPLOAD_FOLDER = 'uploads'

if os.path.exists(UPLOAD_FOLDER):
    shutil.rmtree(UPLOAD_FOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

session_data = {
    "documents": [],
    "indexing_method": "flat"
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("pdfs")
    indexing_method = request.form.get("indexing_method", "flat")
    retreival_method = request.form.get("retreival_method", "MMR")

    saved_filepaths = []
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        saved_filepaths.append(filepath)

    session_data["documents"] = saved_filepaths
    session_data["indexing_method"] = indexing_method
    session_data["retreival_method"] = retreival_method 

    handle_documents_and_qa(saved_filepaths, indexing_method=indexing_method)

    return render_template("chat.html")
 

# @app.route('/chat', methods=['POST'])
# def chat():
#     #clear_vector_db()
#     question = request.form.get("question")
#     indexing_method = session_data.get("indexing_method", "flat")
#     retreival_method = session_data.get("retreival_method", "MMR")

#     answer = answer_question(question=question, indexing_method=indexing_method , retreival_method=retreival_method)

#     return render_template('chat.html', response=answer)

@app.route('/chat', methods=['POST'])
def chat():
    question = request.form.get("question")
    indexing_method = session_data.get("indexing_method", "flat")
    retreival_method = session_data.get("retreival_method", "MMR")

    answer = answer_question(question=question, indexing_method=indexing_method , retreival_method=retreival_method)

    # Store history in session or a global object — for now, let's use a simple global list
    if "chat_history" not in session_data:
        session_data["chat_history"] = []
    session_data["chat_history"].append((question, answer))

    return render_template('chat.html', chat_history=session_data["chat_history"])


if __name__ == '__main__':
    app.run(debug=True)
