#!/usr/bin/env python3
"""
This package is part of an initiative to implement the paradigm 
of multiple agents or LLM components, designed to handle requests
and align them with specific objectives. This node tries to use the best
practices in ROS programming.
Author: Elio David Triana Rodriguez
Universidade Federal do Espirito Santo
Human Centered Systems Laboratory
"""

# Library import section
import rclpy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn
# Libraries for action control
from rclpy.action import ActionServer
from rclpy.action import GoalResponse
from rclpy.action import CancelResponse
# Librerias de interfaces de ROS2
from smart_walker_ai_msgs.action import LlmContextual
from smart_walker_ai_msgs.msg import LlmGoalPose
from smart_walker_ai_msgs.msg import LlmResponse
# Libreries for llm configuration
from langchain_ollama import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Libraries for agent messages configuration
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# Libraries for graph agent configuration
from langgraph.graph import START
from langgraph.graph import END
from langgraph.graph import StateGraph
from typing import Literal
# Libraries for rag execution
from smart_walker_ai.lcad_retrieval.lcad_retriever import SentenceRetriever
from smart_walker_ai.utils.ollama_functions import is_model_available_locally
# Utility library import section
from yaml import safe_load as yaml_load
from time import time
# Self-Library import section
from smart_walker_ai import LocalizationState

# Main node class
class NavigateActionSW(LifecycleNode):
    def __init__(self) -> None:
        super().__init__('navigate_action')

        # Declaration of internal node parameters
        self.declare_parameter("llm_id", "qwen2.5:32b")
        self.declare_parameter("encoder_model_id", "nomic-embed-text")
        self.declare_parameter("temp", 0.0)
        self.declare_parameter("llm_path_configuration", "./src/VWalker/smart_walker_ai/agent_config/agent_configuration.yaml")
        self.declare_parameter("map_path_structured_description", "./src/VWalker/smart_walker_ai/config/StructuredDescription.json")
        self.declare_parameter("map_path_semantic_description", "./src/VWalker/smart_walker_ai/config/")
        self.declare_parameter("chunk_repo_path", "./src/VWalker/smart_walker_ai/retriever_data/pdf/")
        self.declare_parameter("chunk_json_path", "./src/VWalker/smart_walker_ai/retriever_data/json/")

        # Initial node validation
        self.get_logger().info(f"Initial config node. Node-Name: {self.get_name()}")
    
    def on_configure(self, state):
        """
        Configure the node with initial parameters and validate file paths.

        Args:
            state (LifecycleState): The current state of the node.

        Returns:
            TransitionCallbackReturn: SUCCESS if configuration is successful, FAILURE otherwise.
        """
        self.get_logger().info(f"On Configure Node: {self.get_name()}")
        
        # Initialization and assignment of input parameter values
        self.llm_id = self.get_parameter("llm_id").get_parameter_value().string_value
        self.encoder_model_id = self.get_parameter("encoder_model_id").get_parameter_value().string_value
        self.temp = self.get_parameter("temp").get_parameter_value().double_value
        
        # Validacion de existencia de modelo llamado
        if not is_model_available_locally(self.llm_id):
            self.get_logger().error(f"Error, Model called '{self.llm_id}' no existing.")
            return TransitionCallbackReturn.FAILURE

        # Validation of file path existence
        try:
            path_temp = self.get_parameter("llm_path_configuration").get_parameter_value().string_value
            with open(path_temp) as stream:
                self.system_config = yaml_load(stream)
        except Exception as e:
            # None value initialization
            self.system_config = None
            self.get_logger().error(f"File error reading. {e}")
            return TransitionCallbackReturn.FAILURE
        self.map_path_semantic_description = self.get_parameter("map_path_semantic_description").get_parameter_value().string_value
        self.chunk_repo_path = self.get_parameter("chunk_repo_path").get_parameter_value().string_value

        # Validacion de rag de informacion no estructurada
        self.pdf_retriever, RAG_ERROR = RAG_document(
            dir_path = self.map_path_semantic_description,
            chunk_repo_path = self.chunk_repo_path,
            encoder_model = self.encoder_model_id
        )
        if self.pdf_retriever is None:
            self.get_logger().error(RAG_ERROR)
            return TransitionCallbackReturn.FAILURE
        
        self.map_path_structured_description = self.get_parameter("map_path_structured_description").get_parameter_value().string_value
        self.chunk_json_path = self.get_parameter("chunk_json_path").get_parameter_value().string_value

        # Validacion de rag de informacion estructurada
        self.json_retriever, RAG_ERROR = RAG_json(
            file_path = self.map_path_structured_description,
            chunk_repo_path = self.chunk_json_path,
            encoder_model = self.encoder_model_id
        )
        if self.json_retriever is None:
            self.get_logger().error(RAG_ERROR)
            return TransitionCallbackReturn.FAILURE
        self.get_logger().info(f"Configurate Node Successful: {self.get_name()}")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """
        Activate the node.

        Args:
            state (LifecycleState): The current state of the node.

        Returns:
            TransitionCallbackReturn: SUCCESS if activation is successful.
        """
        self.get_logger().info(f"On Activate Node: {self.get_name()}")

        # Ollama connection section
        # Model Calback configuration
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler])
        # Model parameter configuration and connection
        try:
            self.model_connection = ChatOllama(
                model = self.llm_id,
                temperature = self.temp,
                callback_manager = callback_manager,
                num_predict = 512,
                seed = None
            )
        except Exception as e:
            self.get_logger().error(f"Error. Can't configurate the Node. {e}")
            return TransitionCallbackReturn.FAILURE
        # End Ollama connection section

        # Llm agent configuration section
        # Evaluator Agent configuration
        self.evaluator_agent = self.agent_config(
            template_config=self.system_config['agent_available_system_descriptions']['evaluator_agent']["prompt_system"],
            input_vars=self.system_config['agent_available_system_descriptions']['evaluator_agent']["input_values"]
        )
        # PDF query Agent Configuration
        self.pdf_query_agent = self.agent_config(
            template_config=self.system_config['agent_available_system_descriptions']['PDF_query_agent']["prompt_system"],
            input_vars=self.system_config['agent_available_system_descriptions']['PDF_query_agent']["input_values"]
        )

        # Docs Evaluator Agent Configuration
        self.docs_evaluator_agent = self.agent_config(
            template_config=self.system_config['agent_available_system_descriptions']['Docs_evaluator_agent']["prompt_system"],
            input_vars=self.system_config['agent_available_system_descriptions']['Docs_evaluator_agent']["input_values"]
        )

        # Response CreatorAgent Configuration
        self.response_creator_agent = self.agent_config(
            template_config=self.system_config['agent_available_system_descriptions']['response_creator_agent']["prompt_system"],
            input_vars=self.system_config['agent_available_system_descriptions']['response_creator_agent']["input_values"]
        )

        # Bad Response creator  Agent Configuration
        self.bad_response_agent = self.agent_config(
            template_config=self.system_config['agent_available_system_descriptions']['bad_response_agent']["prompt_system"],
            input_vars=self.system_config['agent_available_system_descriptions']['bad_response_agent']["input_values"]
        )
        # Ends Llm agent configuration section

        # Seccion de cracion de ciclo de multiples agentes.
        self.multiAgent_graph = self.multiagent_graph_creation()
        
        # Action server configuration and excecution
        try:
            self.get_logger().info("Try action create")
            self.multi_agent_contextual_action = ActionServer(
                self,
                LlmContextual,
                'multi_agent_contextual_action',
                execute_callback = self.llm_execute_cb,
                goal_callback = self.goal_llm_callback,
                handle_accepted_callback = self.accepted_llm_callback,
                cancel_callback = self.cancel_llm_callback
            )
        except Exception as e:
            self.get_logger().error(f"Error in action creation. {e}")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info(f"Activate Node Successful: {self.get_name()}")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """
        Deactivate the node.

        Args:
            state (LifecycleState): The current state of the node.

        Returns:
            TransitionCallbackReturn: SUCCESS if deactivation is successful.
        """
        self.get_logger().info(f"On Deactivate {self.get_name()}")

        # Deleting active instance variables
        del self.evaluator_agent
        del self.pdf_query_agent
        del self.docs_evaluator_agent
        del self.response_creator_agent
        del self.bad_response_agent
        # 
        del self.multiAgent_graph

        if self.multi_agent_contextual_action:
            self.multi_agent_contextual_action.destroy()

        del self.model_connection

        self.get_logger().info(f"Node Deactivate Successful {self.get_name()}")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """
        Clean up the node.

        Args:
            state (LifecycleState): The current state of the node.

        Returns:
            TransitionCallbackReturn: SUCCESS if cleanup is successful.
        """
        self.get_logger().info(f"Cleaning up {self.get_name()}")
        
        del self.llm_id
        del self.encoder_model_id
        del self.temp
        del self.system_config
        del self.map_path_semantic_description
        del self.chunk_repo_path
        del self.pdf_retriever
        del self.map_path_structured_description
        del self.chunk_json_path
        del self.json_retriever

        self.get_logger().info("Clean Node Successful")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state):
        """
        Shut down the node.

        Args:
            state (LifecycleState): The current state of the node.

        Returns:
            TransitionCallbackReturn: SUCCESS if shutdown is successful.
        """
        self.get_logger().info('Shutting down...')
        return TransitionCallbackReturn.SUCCESS

    def on_error(self, state):
        """
        Handle node errors.

        Args:
            state (LifecycleState): The current state of the node.

        Returns:
            TransitionCallbackReturn: SUCCESS if error handling is successful.
        """
        self.get_logger().info('Error...')
        return TransitionCallbackReturn.SUCCESS
    
    # Class function instances and utility functions
    def agent_config(self, template_config: str, input_vars: list):
        # Funcion para la configuracion de cada agente de llm
        prompt_template = PromptTemplate(
            template = template_config,
            input_variables = input_vars
        )
        chain_llm = prompt_template | self.model_connection | JsonOutputParser()
        return chain_llm
    
    # Class function for multiagent  graph creation
    def multiagent_graph_creation(self):
        def validator_state(state: LocalizationState) -> LocalizationState:
            # Get response from evaluator agent
            print("====================================")
            print(">>Validation Request Node. Passing.")
            print("====================================")
            response = self.evaluator_agent.invoke({"request": state["original_request"]})
            print(response)
            return {
                "validate_request": response["validate_action"]=="yes",
                "rewrite_request": response["direct_request"],
                "map_rag_information": "",
                "pdf_rag_validation": False,
                "pdf_rag_information": "",
                "bad_request_val":False
            }

        def validator_router(state: LocalizationState) -> Literal["map_rag", "bad_request"]:
            print("====================================")
            print(">>validation Router")
            print("====================================")
            if state["validate_request"]:
                return "map_rag"
            else:
                return "bad_request"

        # State for map rag
        def map_rag_state(state: LocalizationState) -> LocalizationState:
            print("====================================")
            print(">>Map Rag State Node. Passing.")
            print("====================================")

            if state["pdf_rag_validation"]:
                tempjson_value = self.json_retriever.get_context(state["pdf_rag_information"], top_k=1)
            else:
                tempjson_value = self.json_retriever.get_context(state["rewrite_request"], top_k=2)
                tempjson_value += self.json_retriever.get_context(state["original_request"], top_k=2)
            print(tempjson_value)

            return {
                "map_rag_information": tempjson_value.copy()
            }

        # Second validation
        def info_validation_state(state: LocalizationState) -> LocalizationState:
            print("====================================")
            print(">>Validation information Node. Passing.")
            print("====================================")
            return state

        def info_validation_router(state: LocalizationState) -> Literal["doc_rag", "bad_request", "continue"]:
            print("====================================")
            print(">>Validation information Router")
            print("====================================")
            response = self.docs_evaluator_agent.invoke({
                "map": state["map_rag_information"],
                "context": state.get("pdf_rag_context", None),
                "original_request": state["original_request"],
                "rewrite_request": state["rewrite_request"]
            })
            print(state)
            response = dict(response)
            response["validate_map_information"] = response.pop(list(response.keys())[0])
            print(response)
            if response['validate_map_information'] == "yes":
                return "continue"
            if not state["pdf_rag_validation"]:
                return "doc_rag"
            return "bad_request"

        # doc pdf rag
        def pdf_rag_state(state: LocalizationState) -> LocalizationState:
            print("====================================")
            print(">>pdf Rag State Node. Passing.")
            print("====================================")
            original_request_retrieval = self.pdf_retriever.get_context(state['original_request'], top_k=3)
            processed_request_retrieval = self.pdf_retriever.get_context(state['rewrite_request'], top_k=3)
            context_value = "\n\nRetrieved Context Fragment: \n\n".join(list(set(original_request_retrieval + processed_request_retrieval)))

            response = self.pdf_query_agent.invoke({
                "original_request": state["original_request"],
                "rewrite_request": state["rewrite_request"],
                "document": context_value
            })

            return {
                "pdf_rag_information": response["name"],
                "pdf_rag_validation": True,
                "pdf_rag_context": context_value
            }

        # Bad request state node
        def bad_request_state(state: LocalizationState) -> LocalizationState:
            print("====================================")
            print(">> Bad request state  Node. Passing.")
            print("====================================")

            return {
                "bad_request_val": True
            } 

        # Final response state
        def final_response_state(state: LocalizationState) -> LocalizationState:
            print("====================================")
            print(">>Final Response Node. Passing.")
            print("====================================")
            # final response invoke
            if not state['bad_request_val']:
                final_response = self.response_creator_agent.invoke({
                    "map": state["map_rag_information"],
                    "document": state["pdf_rag_information"],
                    "request": state["rewrite_request"]
                })
                print(final_response)
            else:
                final_response = self.bad_response_agent.invoke({
                    "map": state["map_rag_information"],
                    "document": state["pdf_rag_information"],
                    "validate_request": state["validate_request"],
                    "request": state["original_request"]
                })

            state['final_coordinate_response'] = final_response
            return state
        
        # Fin de configuracion de nodos de ciclo de funcionamiento

        # Inicio de configuracion de ciclo general.
        graph = StateGraph(LocalizationState)
        graph.add_node("validator_state", validator_state)
        graph.add_node("map_rag_state", map_rag_state)
        graph.add_node("info_validation_state", info_validation_state)
        graph.add_node("pdf_rag_state", pdf_rag_state)
        graph.add_node("bad_request_state", bad_request_state)
        graph.add_node("final_response_state", final_response_state)
        graph.add_edge(START, "validator_state")
        graph.add_conditional_edges(
            "validator_state", validator_router,
            {"map_rag": "map_rag_state", "bad_request": "bad_request_state"}
        )
        graph.add_edge("map_rag_state", "info_validation_state")
        graph.add_conditional_edges(
            "info_validation_state", info_validation_router,
            {"doc_rag": "pdf_rag_state", "bad_request": "bad_request_state", "continue": "final_response_state"}
        )
        graph.add_edge("pdf_rag_state", "map_rag_state")
        graph.add_edge("bad_request_state", "final_response_state")
        graph.add_edge("final_response_state", END)
        cognitive_cycle = graph.compile()
        
        return cognitive_cycle

    # Action configuration section
    def goal_llm_callback(self, goal_request):
        self.get_logger().info('Request received successfully.\n Sending for processing:\n {0}'.format(goal_request.prompt))
        return GoalResponse.ACCEPT
    
    # Callback for successful accepted request action
    def accepted_llm_callback(self, goal_handle):
        self.get_logger().info('Goal accepted and starting execution.')
        # Inicializar hilo de ejecucion
        return goal_handle.execute()
    
    # Callback for successful cancel request action
    def cancel_llm_callback(self, goal_handle):
        self.get_logger().info('Request to cancel received successfully.')
        return CancelResponse.ACCEPT
    
    # Callback principal function from action
    def llm_execute_cb(self, goal_handle):
        # Variables called from action request
        user_request : str = str(goal_handle.request.prompt)

        # Excecution action validation
        try:
            self.get_logger().info("Multiple agents move request.")

            start_time = time()
            # Feedback message variable instance
            feedback_msg = LlmContextual.Feedback()

            # Status from actual response
            feedback_msg.status = "processing request"

            self.get_logger().info("First invokation of multiagents")
            # Inicializacion de primer estado de ejecucion
            initial_state_ = {"original_request": user_request}
            
            for event in self.multiAgent_graph.stream(initial_state_, stream_mode = "updates"):
                # Validacion de estado de procesamiento
                print("Nombre de evento ",event)
                try:
                    # Agent name asignation
                    feedback_msg.agent_name = next(iter(event)).capitalize()
                    # Feedback response
                    feedback_msg.feedback_response = str(event[list(event.keys())[0]])
                    # iter final response
                    final_response = event[list(event.keys())[0]]
                    self.get_logger().info(f"Feedback published.")
                    self.get_logger().info(f"Request Status: {feedback_msg.status}")
                    self.get_logger().info(f"Process Agents: {feedback_msg.agent_name}")
                    self.get_logger().info(f"Actual Message: {feedback_msg.feedback_response}")
                    
                    # Feedback response using
                    goal_handle.publish_feedback(feedback_msg)
                    self.get_logger().debug('Feedback published.')
                except Exception as e:
                    self.get_logger().error(f"Error en solicitud de ejecucion en stream. {e}")
                
        except Exception as e:
            self.get_logger().error(f"An error occurred during streaming: {e}")
            goal_handle.abort()
            result = LlmContextual.Result()
            return result
        
        goal_handle.publish_feedback(feedback_msg)
        self.get_logger().debug('Final feedback published.')
        # actualizar estado completado de objetivo
        goal_handle.succeed()
        # Obtener resultado final
        response_temp = LlmResponse()
        response_temp.user_request = final_response["original_request"]
        response_temp.validate_request = not final_response["bad_request_val"]
        if response_temp.validate_request:
            response_temp.direct_request = final_response["rewrite_request"]
            response_temp.goal_name = final_response["final_coordinate_response"]["name"]
            response_temp.goal_coordinates.x = final_response["final_coordinate_response"]["x"]
            response_temp.goal_coordinates.y = final_response["final_coordinate_response"]["y"]
        response_temp.goal_coordinates.status = "Proccess Complete"
        
        result = LlmContextual.Result()
        result.response = response_temp

        # Log de tiempo de ejecucion de proceso
        end_time = time()
        duration = end_time - start_time
        self.get_logger().info('Goal completed in {0:.2f} seconds.'.format(duration))

        return result
    # End Action configuration section

# Seccion de funciones externas de clase
# Funcion para ejecucion de rag, de informacion no estructurada
def RAG_document(dir_path: str, chunk_repo_path: str, encoder_model: str):
    try:
        pdf_retriever = SentenceRetriever(use_ollama=True, encoder_model_id=encoder_model)
        pdf_retriever.init_pdf_retriever(
            input_dir = dir_path, 
            data_dir = chunk_repo_path, 
            chunk_size = 500,
            overlap_chunks = 50,
            force_rebuild_index = True
        )
        error_text = None
    except Exception as e:
        pdf_retriever = None
        error_text = f"Error while initializing RAG document retriever: {e}"
    return (pdf_retriever, error_text)

def RAG_json(file_path: str, chunk_repo_path: str, encoder_model: str):
    try:
        json_retriever = SentenceRetriever(use_ollama=True, encoder_model_id=encoder_model)
        json_retriever.init_json_retriever(
            json_file_path = file_path, 
            json_key_path = ["area_layout", "offices"],
            json_indexing_keys = ["name"],
            data_dir = chunk_repo_path,
            force_rebuild_index = True
        )
        error_text = None
    except Exception as e:
        json_retriever = None
        error_text = f"Error while initializing RAG-Json document retriever: {e}"
    return json_retriever, error_text

def clean_shutdown(node, executor):
    """
    Ensure a clean termination of processes executed by a specific node.

    Args:
        node: The node to be shut down.
        executor: The executor managing the node.
    """
    try:
        executor.remove_node(node)
        node.destroy_node()
    except Exception as e:
        print(f"Error while removing or destroying node: {e}")

    try:
        executor.shutdown()
    except Exception as e:
        print(f"Error while shutting down executor: {e}")

    try:
        rclpy.shutdown()
    except Exception as e:
        pass

# Main function for node execution
def main():
    """
    Main function to initialize and run the node.
    """
    # Node initialization and assignment
    rclpy.init()
    node = NavigateActionSW()
    
    # Automatic node configuration and activation
    node.trigger_configure()
    node.trigger_activate()

    # Node executor
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Shutting down.")
    finally:
        clean_shutdown(node, executor)

if __name__ == '__main__':
    main()