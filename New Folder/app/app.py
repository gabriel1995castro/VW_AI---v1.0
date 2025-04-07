from flask import Flask
from flask import jsonify
from flask import request
from flask import render_template
from utils.ollama_functions import crient_ollama, agentCreator
from langchain_core.messages import HumanMessage


app = Flask(__name__)


@app.route('/')
def home_validation():
    return jsonify({
        "message": "servidor inicializado"
    })

@app.route('/home')
def home():
    return render_template("home.html", title="homepage")

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get JSON data(
    data = request.get_json()
    
    # Ejemplo
    msg = data["message"]

    #response = client.invoke(message)
    agent_ = agentCreator()
    try:
        response = agent_.invoke({"messages": HumanMessage(content=msg)})["messages"]
        response = response[-1]
        return jsonify({
        "message": response.content
            #"message": "TESTE"
        })
    except Exception as e:
        response = {"content" : f"Error trying to call the model. {e}"}
        return jsonify({
            "message": response['content']
            #"message": "TESTE"
        })
    #print("hola")
    

if __name__=="__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)