import os  
from glob import glob
from setuptools import find_packages, setup

package_name = 'face_recog'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.xml')),
        (os.path.join('lib', package_name), glob('face_recog/classifier_model.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='paul',
    maintainer_email='33916898+elpidiovaldez@users.noreply.github.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'live_face_recog = face_recog.live_face_recog:main',
        ],
    },
)
