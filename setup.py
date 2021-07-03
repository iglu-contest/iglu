from operator import methodcaller
from setuptools import setup, find_packages
from os.path import join, dirname


with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name='iglu',
    version='0.2.0',
    include_package_data=True,
    packages=find_packages(exclude=['test', 'test.*']),
    long_description='',
    install_requires='\n'.join([
        'minerl_patched', requirements
    ]),
    dependency_links=[
        'https://github.com/iglu-contest/minerl/releases/download/v0.3.7-patched/minerl_patched-0.4.0.zip',
    ]
)
