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

last = None

@app.route('/chat', methods=['POST'])
def chat():

    global last
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    # Get JSON data(
    data = request.get_json()
        
    # Ejemplo
    msg = data["message"]

    #response = client.invoke(message)
    agent_ = agentCreator()
    try:
        tools_used = []
        response = agent_.invoke({"messages": HumanMessage(content=msg)})["messages"]
        response_llm = response[-1]
        
        for item in response:
            # Verifica se é formato objeto com 'tool_calls'
            if hasattr(item, 'tool_calls') and item.tool_calls:
                for tool in item.tool_calls:
                    # Verifica se é objeto com .function
                    if hasattr(tool, 'function'):
                        tools_used.append(tool.function.name)
                    # Ou se é dict
                    elif isinstance(tool, dict) and 'name' in tool:
                        tools_used.append(tool['name'])

            # Verifica se há tool_calls no additional_kwargs
            elif hasattr(item, 'additional_kwargs') and 'tool_calls' in item.additional_kwargs:
                for tool in item.additional_kwargs['tool_calls']:
                    if isinstance(tool, dict) and 'name' in tool:
                        tools_used.append(tool['name'])

        print (tools_used)

        if "activate_free_navigation" in tools_used:
            actual_mode = "manual"
            last = actual_mode

        elif "activate_assisted_navigation" in tools_used:
            actual_mode = "assist"
            last = actual_mode
        
        if not tools_used:
            actual_mode = last


        return jsonify({
        "message": response_llm.content,
        "mode": actual_mode
            #"message": "TESTE"
        })
    except Exception as e:
        response = {"content" : f"Error trying to call the model. {e}"}
        return jsonify({
            "message": response['content'],
            "mode":None
            #"message": "TESTE"
        })
    #print("hola")
    


if __name__=="__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', debug=True)