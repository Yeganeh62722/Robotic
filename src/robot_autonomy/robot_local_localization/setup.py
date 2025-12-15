from setuptools import setup

package_name = 'robot_local_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name], # Tells setuptools to find the folder named 'robot_local_localization'
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='ROS 2 package for localization prediction updates.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'prediction_node = robot_local_localization.prediction_node:main',
            'measurement_node = robot_local_localization.measurement_node:main',
            'ekf_node = robot_local_localization.ekf_node:main',
            'test_node = robot_local_localization.test_node:main',
        ],
    },
)