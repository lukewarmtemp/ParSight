from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to mavros.launch.py from px4_autonomy_modules package
    mavros_launch_path = os.path.join(
        get_package_share_directory('px4_autonomy_modules'),  # Package name
        'launch',                                           # Folder name
        'mavros.launch.py'                                  # Launch file name
    )

    # Include mavros.launch.py
    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(mavros_launch_path),
    )

    return LaunchDescription([
        mavros_launch,
    ])
