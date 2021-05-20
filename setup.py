from setuptools import setup, find_packages
from os.path import join, dirname


with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name='iglu',
    version='0.1',
    packages=find_packages(exclude=['test', 'test.*']),
    long_description='',
    install_requires=[
        'minerl'
    ],
    dependency_links=[
        'lib/minerl-0.4.0-patched.tar.gz',
    ]
)