#!/usr/bin/env python3
"""
ManusMocap node that allows setting the ROS node name (useful for running
left/right instances in the same process or in the same launch file).
"""
import argparse
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseArray


class ManusMocap(Node):
    """
    ROS2 node that subscribes to a PoseArray topic and provides live mocap
    point data as NumPy arrays.

    Args:
        node_name (str): ROS node name to use (default: 'realtime_mocap').
        topic_name (str): PoseArray topic to subscribe to (default: '/manus_poses').
        queue_size (int): subscription queue size / QoS depth (default: 10).
    """
    def __init__(self,
                 node_name: str = 'realtime_mocap',
                 topic_name: str = '/manus_poses',
                 queue_size: int = 10):
        # IMPORTANT: pass node_name to super().__init__ so different instances
        # can have different ROS node names (e.g., 'manus_mocap_right', 'manus_mocap_left')
        super().__init__(node_name)

        # storage for the latest frame of mocap points
        self.human_points: np.ndarray | None = None

        # subscribe to the PoseArray topic
        self.subscription = self.create_subscription(
            PoseArray,
            topic_name,
            self._pose_callback,
            queue_size
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f"Node '{node_name}' subscribed to {topic_name}")

    def _pose_callback(self, msg: PoseArray) -> None:
        # convert PoseArray into a Nx3 NumPy array
        n = len(msg.poses)
        arr = np.zeros((n, 3), dtype=float)
        for i, pose in enumerate(msg.poses):
            arr[i, 0] = pose.position.x
            arr[i, 1] = pose.position.y
            arr[i, 2] = pose.position.z
        self.human_points = arr
        self.get_logger().debug(f"Received frame with {n} points")

    def get(self) -> dict:
        """
        Retrieve the latest mocap frame.

        Returns:
            A dict:
              - 'result': Nx3 NumPy array of point positions, or None if no data yet
              - 'status': 'streaming' if data available, else 'no_data'
        """
        if self.human_points is None:
            return {"result": None, "status": "no_data"}
        return {"result": self.human_points.copy(), "status": "streaming"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-name', type=str, default='realtime_mocap',
                        help="ROS node name to use (e.g., 'manus_mocap_right')")
    parser.add_argument('--topic-name', type=str, default='/manus_poses',
                        help="PoseArray topic to subscribe to (e.g., '/manus_poses_right')")
    parser.add_argument('--queue-size', type=int, default=10, help='Subscription queue size')
    args = parser.parse_args(argv)

    rclpy.init()
    mocap = ManusMocap(node_name=args.node_name, topic_name=args.topic_name, queue_size=args.queue_size)
    try:
        rclpy.spin(mocap)
    except KeyboardInterrupt:
        pass
    finally:
        mocap.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
