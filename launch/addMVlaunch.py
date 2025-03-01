import os
import sys
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    from launch_ros.descriptions import ComposableNode
    from launch_ros.actions import ComposableNodeContainer, Node
    from launch.actions import TimerAction, Shutdown
    from launch import LaunchDescription
    
    rune_composable_node = Node(
        package='yolo_pose_buff',
        executable='yolo_pose_buff',
        name='rm_rune',
        output='both',
        emulate_tty=True,
        on_exit=Shutdown(),
    ) 

    return LaunchDescription([
       rune_composable_node
    ])
