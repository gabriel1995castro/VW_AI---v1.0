#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from smart_walker_ai_msgs.action import LlmContextual
from nav2_msgs.action import ComputePathToPose
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger 
from std_msgs.msg import String 

class LlmActionClient(Node):
    def __init__(self):
        super().__init__('llm_action_client')

        self.new_goal = None
        self.client = ActionClient(self, LlmContextual, 'multi_agent_contextual_action')
        self.compute_path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose') 
        self.compute_path_client.wait_for_server()
        self.path_pub = self.create_publisher(Path, 'calculated_path', 10)
        self.reload_path_client = self.create_client(Trigger, 'reload_path')
        self.subscription = self.create_subscription(String,'user_command',self.command_callback,10)
        

        self.current_future = None
    
    def generate_path(self, goal_x, goal_y):
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = "map"
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = goal_x
        goal_msg.goal.pose.position.y = goal_y
        goal_msg.goal.pose.orientation.w = 1.0  

        future_mov = self.compute_path_client.send_goal_async(goal_msg)
        future_mov.add_done_callback(self.on_goal_completed)

    def on_goal_completed(self, future):
        try:
            goal_handle = future.result()
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.on_result_received)
        except Exception as e:
            self.get_logger().error(f"Erro ao calcular o caminho: {e}")

    def on_result_received(self, future):
        try:
            result = future.result()
            path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in result.result.path.poses]      
            self.get_logger().info(f'Caminho gerado: {path_points}')

            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses = result.result.path.poses

            self.path_pub.publish(path_msg)
            self.get_logger().info(f'Caminho final: {path_msg.poses[-1].pose}')
            self.call_reload_service()

        except Exception as e:
            self.get_logger().error(f"Erro ao processar o resultado: {e}")

        self.current_future = None 

    def call_reload_service(self):
        """ Envia uma requisição para o serviço 'reload_path' para atualizar o caminho """
        if not self.reload_path_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Serviço 'reload_path' não está disponível.")
            return

        req = Trigger.Request()
        future = self.reload_path_client.call_async(req)
        future.add_done_callback(self.reload_service_callback)

    def reload_service_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Serviço 'reload_path' chamado com sucesso: {response.message}")
            else:
                self.get_logger().error("Falha ao chamar o serviço 'reload_path'.")
        except Exception as e:
            self.get_logger().error(f"Erro ao chamar o serviço: {e}")

    def command_callback(self, msg):

        self.get_logger().info(f'Received user command: {msg.data}')
        self.send_goal(msg.data)  

    def send_goal(self, prompt_text):
        goal_msg = LlmContextual.Goal()
        goal_msg.prompt = prompt_text

        self.get_logger().info(f'Sending goal: {goal_msg.prompt}')
        
        self.client.wait_for_server()
        self._send_goal_future = self.client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.status} | Agent: {feedback_msg.feedback.agent_name} | Response: {feedback_msg.feedback.feedback_response}')

    def get_result_callback(self, future):
        result = future.result().result
        
        if not result.response.validate_request:
            self.get_logger().info("Invalid request, not publishing coordinates.")
            return

        self.get_logger().info(f'Final Response: {result.response.user_request}')
        self.get_logger().info(f'Goal Coordinates: x={result.response.goal_coordinates.x}, y={result.response.goal_coordinates.y}')
        
        pose_msg = Pose()
        pose_msg.position.x = float(result.response.goal_coordinates.x)
        pose_msg.position.y = float(result.response.goal_coordinates.y)
        self.new_goal = future.result().result
        self.generate_path(pose_msg.position.x, pose_msg.position.y)
        
        self.get_logger().info(f'Publishing Pose: x={pose_msg.position.x}, y={pose_msg.position.y}')
        self.get_logger().info("Pose published successfully!")

    """def send_goal(self, order):
        # Wait for the action server to be available
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create the goal
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order


        # Send the goal and get the future response
        self.get_logger().info(f'Sending goal with order: {order}')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, self.get_result_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
       """ 

def main():
    rclpy.init()
    action_client = LlmActionClient()
    #action_client.send_goal("I need go to human centered laboratory.")  
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
