#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose
from smart_walker_ai_msgs.action import LlmContextual
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path  

class LlmActionClient(Node):
    def __init__(self):
        super().__init__('llm_action_client')

        # Cliente da action
        self.client = ActionClient(self, LlmContextual, 'multi_agent_contextual_action')
        self.compute_path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose') 
        self.compute_path_client.wait_for_server()

        # Publisher para publicar a resposta como Pose
        self.path_pub = self.create_publisher(Path, 'calculated_path', 10)
        self.current_future = None
    
    def generate_path(self, goal_x, goal_y):
        # Envia um objetivo para calcular o caminho
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal = PoseStamped()
        goal_msg.goal.header.frame_id = "map"
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = goal_x
        goal_msg.goal.pose.position.y = goal_y
        goal_msg.goal.pose.orientation.w = 1.0  

        # Enviando o objetivo de forma assíncrona para o cálculo do caminho
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

            # Publica o caminho gerado
            path_msg = Path()
            path_msg.header.frame_id = "map"
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for pose in result.result.path.poses:
                path_msg.poses.append(pose)

            self.path_pub.publish(path_msg)
            self.get_logger().info(f'Caminho final: {path_msg.poses[-1].pose}')
        
        except Exception as e:
            self.get_logger().error(f"Erro ao processar o resultado: {e}")

        self.current_future = None 

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
        
        # Criando a mensagem de pose
        pose_msg = Pose()
        pose_msg.position.x = float(result.response.goal_coordinates.x)
        pose_msg.position.y = float(result.response.goal_coordinates.y)

        # Chamando a função de gerar o caminho imediatamente
        self.generate_path(pose_msg.position.x, pose_msg.position.y)
        
        self.get_logger().info(f'Publishing Pose: x={pose_msg.position.x}, y={pose_msg.position.y}')
        self.get_logger().info("Pose published successfully!")

def main():
    rclpy.init()
    action_client = LlmActionClient()
    action_client.send_goal("I need go to human centered laboratory.")  # Substitua pelo prompt desejado
    rclpy.spin(action_client)

if __name__ == '__main__':
    main()
