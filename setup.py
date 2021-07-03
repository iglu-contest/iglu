
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    markdown = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setup(
    name='iglu',
    version='0.2.1',
    description='IGLU: Interactive Grounded Language Understanding in Minecraft',
    long_description=markdown,
    long_description_content_type="text/markdown",
    url='https://github.com/iglu-contest/iglu',
    author='IGLU team',
    author_email='info@iglu-contest.net',
    include_package_data=True,
    packages=find_packages(exclude=['test', 'test.*']),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
