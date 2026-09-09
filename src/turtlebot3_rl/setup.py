import os 
from glob import glob
from setuptools import find_packages, setup

package_name = 'turtlebot3_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	(os.path.join('share', package_name , 'launch'),glob('launch/*.launch.py')),
	(os.path.join('share', package_name , 'worlds'),glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='landradepro',
    maintainer_email='luis_andrade_p@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'train = turtlebot3_rl.train:main',
		'evaluate = turtlebot3_rl.evaluate:main',
        ],
    },
)
