from setuptools import setup

package_name = 'quadrotor_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    description='Quadrotor model',
    license='MIT',
    entry_points={
        'console_scripts': [
            'quadrotor_node = quadrotor_sim.quadrotor_node:main',
        ],
    },
)
