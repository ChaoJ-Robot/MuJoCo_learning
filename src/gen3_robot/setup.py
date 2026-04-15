from setuptools import setup, find_packages
from glob import glob
import os
import sys

package_name = 'gen3_robot'

if '--editable' in sys.argv:
    sys.argv.remove('--editable')

model_files = []
for root, dirs, files in os.walk('models'):
    for file in files:
        src = os.path.join(root, file)
        dst = os.path.join('share', package_name, root)
        model_files.append((dst, [src]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ] + model_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jack',
    maintainer_email='d13292975990@163.com',
    description='Gen3 robot ROS2 package with MuJoCo',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'gen3_server = gen3_robot.gen3_server:main',
            'command_sender = gen3_robot.command_sender:main',
            'circle_sender = gen3_robot.circle_sender:main',
        ],
    },
)