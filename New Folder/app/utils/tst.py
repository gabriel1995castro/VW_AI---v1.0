import rclpy
from rclpy.node import Node
from typing import Dict, Any
import json
from std_srvs.srv import SetBool
from langchain_core.tools import tool
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from smart_walker_ai_msgs.action import LlmContextual
from nav2_msgs.action import ComputePathToPose
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
from nav_msgs.msg import Path
from std_srvs.srv import Trigger 

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
    request_atu= node.create_client(Trigger, 'reload_path')
    path_pub = node.create_publisher(Path, 'calculated_path', 10)


    # Wait for the action server to be available
    if not action_client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error('Action server not available')
        return "Error: Action server not available"

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
    
    # Get the result
    try:
        result = result_future.result().result.response
        print(result)
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
        path_goal = ComputePathToPose.Goal()
        path_goal.goal = goal_pose
        
        # Send path computation goal
        send_path_goal_future = action_client_nav2.send_goal_async(path_goal)
        rclpy.spin_until_future_complete(node, send_path_goal_future)
        path_goal_handle = send_path_goal_future.result()
        path_pub.publish(path_goal_handle)
        
        if not path_goal_handle.accepted:
            node.get_logger().error('Path computation goal was rejected')
            return "Error: Path computation goal rejected"
        
        # Get path computation result
        path_result_future = path_goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, path_result_future)

        if result.validate_request:
            return f"Path generated to: {result.goal_name}"
        else:
            return f"Thats not possible generate the path for the requested place."
        # Process and return the result
        # Adjust this based on your actual result structure
        
    except Exception as e:
        print(f"Error obtenido en {e}")
        return f"Error trying to get response. {e}"
    