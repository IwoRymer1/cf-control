from setuptools import setup

package_name = 'quadrotor_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='developer@example.com',
    description='Quadrotor simulation controllers',
    license='TODO',
    entry_points={
        'console_scripts': [
            'lee_gazebo_controller = quadrotor_sim.lee_gazebo_controller:main',
        ],
    },
)
