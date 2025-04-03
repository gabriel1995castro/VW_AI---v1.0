import rclpy
from rclpy.node import Node
from typing import Dict, Any
import json
from std_srvs.srv import SetBool
from langchain_core.tools import tool
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from smart_walker_ai_msgs.action import LlmContextual
from nav_msgs.msg import Odometry
from nav2_msgs.action import ComputePathToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from nav_msgs.msg import Path
from std_srvs.srv import Trigger


@tool
def activate_assisted_navigation (query: str):
    """
    Review the user's request and use this function if you understand that the user needs help or needs to be guided to a specific location, use this tool.
    Don't forget to inform the currently active navigation mode of the walker.
    """
    rclpy.init ()
    node = rclpy.create_node('assisted_nav_activate')
    
    client = node.create_client(SetBool, "/admittance_modulator/set_validation")
        # Wait for the service to be available
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('Service not available')
        return False

    # Create a request
    request = SetBool.Request()
    request.data = False

    # Send the request and wait for response
    future = client.call_async(request)

    # Spin until we get a response
    rclpy.spin_until_future_complete(node, future)

    try:
    
        response = future.result()
        print(f"Service call successful: {response.success}")
        print(f"Message: {response.message}")
        return f"Assistance model activation successfully: response = {response.success}"
    
    except Exception as e:
        print(f"Service call failed: {str(e)}")
        return f"Assistance model activation NOT successfully: response = {response.success}"
    
    finally:

        node.destroy_node()
        rclpy.shutdown()

    return False

@tool
def activate_free_navigation (query: str):
    """
    Review the user's request and use this function if you understand that the user does not wish to receive assistance.
    Don't forget to inform the currently active navigation mode of the walker. Return True if using this tool.
    """
    rclpy.init ()
    node = rclpy.create_node('free_nav_activate')
    
    client = node.create_client(SetBool, "/admittance_modulator/set_validation")
        # Wait for the service to be available
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error('Service not available')
        return False

    # Create a request
    request = SetBool.Request()
    request.data = True

    # Send the request and wait for response
    future = client.call_async(request)

    # Spin until we get a response
    rclpy.spin_until_future_complete(node, future)

    try:
    
        response = future.result()
        print(f"Service call successful: {response.success}")
        print(f"Message: {response.message}")
        return response.success
    
    except Exception as e:
        print(f"Service call failed: {str(e)}")
        return False
    
    finally:

        node.destroy_node()
        rclpy.shutdown()

    # return { "result": True,
    #          "message": "Free navigation activated.",
    #          "debug_info": query
    #         }

@tool
def create_path_for_navigation(user_request: str):
    """
    This function is used to create path for specific place. Activate the assisted navigation before to use this tool.
    - Include as input parameter "user_request" the precise user request. 
    - If return error, response with specific error returned.
    """
    rclpy.init()

    # Create a temporary node for the action call
    node = Node('coordinates_request')

    # try:
    # Create an action client
    action_client = ActionClient(node, LlmContextual, '/multi_agent_contextual_action')
    action_client_nav2 = ActionClient (node, ComputePathToPose, '/compute_path_to_pose')
    reload_path_client = node.create_client(Trigger, 'reload_path')
    odom_subscription = node.create_subscription(Odometry, '/odom', lambda msg: setattr(node, 'current_pose', msg), 10)
    path_publisher = node.create_publisher(Path, 'calculated_path', 10)
    node.current_pose = None

    # Wait for the action server to be available
    while not action_client_nav2.wait_for_server(timeout_sec=1.0):
        node.get_logger().info("Aguardando servidor de ação 'compute_path_to_pose'...")
    node.get_logger().info("Conectado ao servidor de ação!")


    # Create a goal message
    goal_msg = LlmContextual.Goal()
    
    # Use the available prompt field to pass context
    goal_msg.prompt = str(user_request)

    # Send the goal
    send_goal_future = action_client.send_goal_async(goal_msg)

    # Wait for the goal to be accepted
    rclpy.spin_until_future_complete(node, send_goal_future)
    goal_handle = send_goal_future.result()

    if not goal_handle.accepted:
        node.get_logger().error('Goal was rejected')
        return "Error: Goal was rejected"

    # Wait for the result
    result_future = goal_handle.get_result_async()
    
    rclpy.spin_until_future_complete(node, result_future)
    print ("Cooordenadas obtidas atráves do RAG ....")
    
    # Get the result
    try:

        result = result_future.result().result.response
        
        #Obtenção da localização atual do andador.
        current_x = node.current_pose.pose.pose.position.x
        current_y = node.current_pose.pose.pose.position.y
        
        node.get_logger().info(f"Posição atual: ({current_x}, {current_y})")

       #função para a geração do caminho
              # Prepare goal pose for path computation
        
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = node.get_clock().now().to_msg()
        
        # Set goal coordinates from contextual result
        goal_pose.pose.position.x = result.goal_coordinates.x
        goal_pose.pose.position.y = result.goal_coordinates.y
        goal_pose.pose.orientation.z = float(0.0)
        goal_pose.pose.orientation.w = float(1.0)
        
        
        # Wait for path computation server
        if not action_client_nav2.wait_for_server(timeout_sec=5.0):
            node.get_logger().error('Path computation server not available')
            return "Error: Path computation server unavailable"
        
        # Prepare path computation goal
        goal_msg = ComputePathToPose.Goal()
        goal_msg.start = PoseStamped()
        goal_msg.start.header.frame_id = "map"
        goal_msg.start.pose = node.current_pose.pose.pose
        goal_msg.goal = goal_pose
        
        # Send path computation goal
        send_path_goal_future = action_client_nav2.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(node, send_path_goal_future)
        path_goal_handle = send_path_goal_future.result()
        
        
        if not path_goal_handle.accepted:
            node.get_logger().error('Path computation goal was rejected')
            return "Error: Path computation goal rejected"
        
        # Get path computation result
        path_result_future = path_goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, path_result_future)
        path_result = path_result_future.result()
        computed_path = path_result.result.path
        path_publisher.publish(computed_path)
        
        
        if not reload_path_client.wait_for_service(timeout_sec=2.0):
            node.get_logger().warn('reload_path service not available')
       
        else:
            request = Trigger.Request()
            reload_future = reload_path_client.call_async(request)
            rclpy.spin_until_future_complete(node, reload_future)

        if result.validate_request:
            node.destroy_node()
            rclpy.shutdown()
            return f"Path generated to: {result.goal_name}"
        else:
            return f"Thats not possible generate the path for the requested place."
        # Process and return the result
        # Adjust this based on your actual result structure
        
    except Exception as e:
        print(f"Error obtenido en {e}")
        return f"Error trying to get response. {e}"
    