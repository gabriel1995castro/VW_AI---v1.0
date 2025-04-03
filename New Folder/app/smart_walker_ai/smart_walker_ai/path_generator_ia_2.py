#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose
from smart_walker_ai_msgs.action import LlmContextual
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger 
from std_msgs.msg import String
import threading
import time

class LlmActionClient(Node):
    def __init__(self):
        super().__init__('llm_action_client')
        
        self.client = ActionClient(self, LlmContextual, 'multi_agent_contextual_action')
        self.compute_path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.path_pub = self.create_publisher(Path, 'calculated_path', 10)
        self.reload_path_client = self.create_client(Trigger, 'reload_path')
        self.subscription = self.create_subscription(String, 'user_command', self.command_callback, 10)
        
        
        self.processing_command = False
        self.current_command = None
        self.command_queue = []
        
        self.get_logger().info("Waiting for action servers...")
        self.client.wait_for_server()
        self.compute_path_client.wait_for_server()
        self.get_logger().info("Action servers ready")

        self.command_thread = threading.Thread(target=self.process_command_queue)
        self.command_thread.daemon = True
        self.command_thread.start()

    def process_command_queue(self):
        while rclpy.ok():
            if self.command_queue and not self.processing_command:
                command = self.command_queue.pop(0)
                self.processing_command = True
                self.current_command = command
                self.get_logger().info(f"Processing command: {command}")
                self._process_command(command)
            time.sleep(0.1)

    def _process_command(self, command):
        try:
            goal_msg = LlmContextual.Goal()
            goal_msg.prompt = command
            
            self.get_logger().info(f'Sending goal: {goal_msg.prompt}')
            
            self._send_goal_future = self.client.send_goal_async(
                goal_msg, 
                feedback_callback=self.feedback_callback
            )
            self._send_goal_future.add_done_callback(self.goal_response_callback)
        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
            self.processing_command = False
            self.current_command = None

    def generate_path(self, goal_x, goal_y):
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = "map"
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = goal_x
        goal_msg.goal.pose.position.y = goal_y
        goal_msg.goal.pose.orientation.w = 1.0  

        self.get_logger().info(f"Generating path to ({goal_x}, {goal_y})")
        future_mov = self.compute_path_client.send_goal_async(goal_msg)
        future_mov.add_done_callback(self.on_goal_completed)

    def on_goal_completed(self, future):
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Path computation goal was rejected!")
                self.processing_command = False
                self.current_command = None
                return
                
            self.get_logger().info("Path computation goal accepted")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.on_result_received)
        except Exception as e:
            self.get_logger().error(f"Error calculating path: {e}")
            self.processing_command = False
            self.current_command = None

    def on_result_received(self, future):
        try:
            result = future.result()
            
            if len(result.result.path.poses) > 0:
                path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in result.result.path.poses]      
                self.get_logger().info(f'Path generated with {len(path_points)} points')
                self.get_logger().debug(f'Path details: {path_points}')

                path_msg = Path()
                path_msg.header.frame_id = "map"
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.poses = result.result.path.poses

                self.path_pub.publish(path_msg)
                self.get_logger().info(f'Final path point: ({path_msg.poses[-1].pose.position.x}, {path_msg.poses[-1].pose.position.y})')
                self.call_reload_service()
            else:
                self.get_logger().error("Received empty path")
        except Exception as e:
            self.get_logger().error(f"Error processing path result: {e}")
        finally:
            self.processing_command = False
            self.current_command = None

    def call_reload_service(self):
        """Sends a request to the 'reload_path' service to update the path"""
        if not self.reload_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Service 'reload_path' not available.")
            return

        req = Trigger.Request()
        future = self.reload_path_client.call_async(req)
        future.add_done_callback(self.reload_service_callback)

    def reload_service_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Service 'reload_path' called successfully: {response.message}")
            else:
                self.get_logger().error("Failed to call service 'reload_path'.")
        except Exception as e:
            self.get_logger().error(f"Error calling service: {e}")

    def command_callback(self, msg):
        self.get_logger().info(f'Received user command: {msg.data}')
        self.command_queue.append(msg.data)

    def send_goal(self, prompt_text):
        self.command_queue.append(prompt_text)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            self.processing_command = False
            self.current_command = None
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: {feedback.status} | Agent: {feedback.agent_name} | Response: {feedback.feedback_response}')

    def get_result_callback(self, future):
        result = future.result().result
        
        if not result.response.validate_request:
            self.get_logger().info("Invalid request, not publishing coordinates.")
            self.processing_command = False
            self.current_command = None
            return

        self.get_logger().info(f'Final Response: {result.response.user_request}')
        self.get_logger().info(f'Goal Coordinates: x={result.response.goal_coordinates.x}, y={result.response.goal_coordinates.y}')
        
        try:
            pose_msg = Pose()
            pose_msg.position.x = float(result.response.goal_coordinates.x)
            pose_msg.position.y = float(result.response.goal_coordinates.y)

            self.generate_path(pose_msg.position.x, pose_msg.position.y)
            
            self.get_logger().info(f'Publishing Pose: x={pose_msg.position.x}, y={pose_msg.position.y}')
        except ValueError as e:
            self.get_logger().error(f"Invalid coordinates: {e}")
            self.processing_command = False
            self.current_command = None

def main():
    rclpy.init()
    action_client = LlmActionClient()
    executor = MultiThreadedExecutor()
    executor.add_node(action_client)
    
    try:
        print("Starting path generator...")
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        action_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()