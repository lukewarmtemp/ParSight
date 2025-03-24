from setuptools import setup

package_name = 'ParSight'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetson',
    maintainer_email='felicia.wanjin.liu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vicon = ParSight.perception:main',
            'camera = ParSight.camera:main',
            'segment = ParSight.segment:main',
            'rgb_camera = ParSight.rgb_camera_node:main',
            'yolo_seg = ParSight.yolo_seg_node:main',
            'test_camm = ParSight.test_camera:main'
        ],
    },
)
