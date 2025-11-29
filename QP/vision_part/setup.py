from setuptools import setup

package_name = 'aruco_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='airlab',
    maintainer_email='airlab@example.com',
    description='ArUco marker tracker using D435 camera',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_tracker = aruco_tracker.d435_aruco_marker_tracker:main',
        ],
    },
)
