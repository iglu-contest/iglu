from setuptools import setup, find_packages
from os.path import join, dirname


with open("requirements.txt", "r") as fh:
    requirements = fh.read()

# TODO: configure installation of iglu

setup(
    name='iglu',
    version='0.1',
    include_package_data=True,
    packages=find_packages(exclude=['test', 'test.*']),
    long_description='',
    install_requires=[
        'minerl_patched',
        'pandas'
    ],
    dependency_links=[
        'https://github.com/iglu-contest/minerl/releases/download/v0.3.7-patched/minerl-0.4.0-patched.zip',
    ]
)
