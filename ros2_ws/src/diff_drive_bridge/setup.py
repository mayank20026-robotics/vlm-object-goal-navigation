from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'diff_drive_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics_lab',
    maintainer_email='robotics_lab@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ['diff_drive_bridge = diff_drive_bridge.diff_drive_bridge:main',
                            'noodom_diff_drive_bridge = diff_drive_bridge.noodom_diff_drive_bridge:main',
                            'scan_rotator = diff_drive_bridge.scan_rotator:main',
        ],
    },
)
