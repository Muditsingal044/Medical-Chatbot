from flask import Flask, jsonify, request
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from src.database import log_chat, get_chat_history, get_recent_context, clear_chat_history
import datetime
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import socket

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your-secret-key'  # Change this to a secure secret key

def get_ip():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY=os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot2" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2})

from langchain_groq import ChatGroq

chatModel = ChatGroq(model="llama3-70b-8192")

system_prompt = (
    "You are a medical assistant helping users with health questions. "
    "First, ask one or two relevant questions to clarify symptoms, "
    "then provide medicine suggestions and care steps. "
    "Use simple language and basic simple words, keep answers concise (max 3 sentences), "
    "and say 'I don't know' if the answer is not in the context. "
    "Consider the chat history when providing answers.\n\n"
    "Chat History:\n{chat_history}\n\n"
    "Current Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
).partial(chat_history="")

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/", methods=["GET"])
def index():
    # Clear chat history when the API is accessed fresh
    clear_chat_history()
    return jsonify({
        "status": "success",
        "message": "Medical Chatbot API is running"
    })

@app.route("/api/clear", methods=["POST"])
def clear_history():
    try:
        clear_chat_history()
        return jsonify({
            "status": "success",
            "message": "Chat history cleared"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/history", methods=["GET"])
def history():
    try:
        chat_logs = get_chat_history()
        history_data = [{
            "user_message": log.user_message,
            "bot_response": log.bot_response,
            "timestamp": log.timestamp.isoformat()
        } for log in chat_logs]
        
        return jsonify({
            "status": "success",
            "history": history_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "status": "error",
                "message": "No message provided"
            }), 400

        user_message = data['message']
        
        # Get recent chat history
        recent_logs = get_recent_context(5)  # Get last 5 interactions
        chat_history = ""
        if recent_logs:
            for log in recent_logs:
                chat_history += f"User: {log.user_message}\nAssistant: {log.bot_response}\n\n"
        
        # Create prompt with history
        prompt = {
            "input": user_message,
            "chat_history": chat_history
        }
        
        # Get response with context
        response = rag_chain.invoke(prompt)
        bot_response = response["answer"]
        
        # Save to database silently
        log_chat(user_message, bot_response)
        
        return jsonify({
            "status": "success",
            "response": bot_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500



if __name__ == "__main__":
    host_ip = get_ip()
    print(f"\nServer running at: http://{host_ip}:8080/")
    print("Share this URL with clients on your network to access the chatbot.")
    print("Press CTRL+C to stop the server.\n")
    app.run(host='0.0.0.0', port=8080, debug=True)
