import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient

rclpy.init()
node = Node('coordinates_request')

# Cliente de ação para calcular o caminho
action_client_nav2 = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')

# Inscrição no tópico de odometria
odom_subscription = node.create_subscription(Odometry, '/odom', lambda msg: setattr(node, 'current_pose', msg), 10)

node.current_pose = None

# Variável para controlar se a requisição já foi feita
path_requested = False

# Aguarda o servidor de ação
while not action_client_nav2.wait_for_server(timeout_sec=1.0):
    node.get_logger().info("Aguardando servidor de ação 'compute_path_to_pose'...")
node.get_logger().info("Conectado ao servidor de ação!")

# Loop principal
while rclpy.ok():
    # Verifica se a posição do robô já foi recebida e se a requisição ainda não foi feita
    if node.current_pose is not None and not path_requested:
        current_x = node.current_pose.pose.pose.position.x
        current_y = node.current_pose.pose.pose.position.y
        node.get_logger().info(f"Posição atual: ({current_x}, {current_y})")

        # Definir destino desejado (exemplo)
        goal_x = 1.5
        goal_y = 1.0

        # Criar a mensagem para o objetivo
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = node.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation.w = 1.0

        goal_msg = ComputePathToPose.Goal()
        goal_msg.start = PoseStamped()
        goal_msg.start.header.frame_id = "map"
        goal_msg.start.pose = node.current_pose.pose.pose
        goal_msg.goal = goal_pose

        # Envia a requisição de caminho
        future = action_client_nav2.send_goal_async(goal_msg)
        future.add_done_callback(lambda future: node.get_logger().info("Requisição de caminho enviada!"))

        # Marca que a requisição foi feita
        path_requested = True

        rclpy.spin_once(node, timeout_sec=2.0)  # Espera por 2 segundos após a requisição

    rclpy.spin_once(node, timeout_sec=1.0)  # Espera por 1 segundo enquanto não tem odometria

# Finaliza o nó quando a execução é interrompida
node.destroy_node()
rclpy.shutdown()
