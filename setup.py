# setup.py
from setuptools import setup, find_packages

setup(
    name="agentic_rag",
    version="1.0",
    package_dir={"": "src"}, # Tells setuptools that packages are under the 'src' directory
    packages=find_packages(where="src"), # Tells it to look for packages inside 'src'
)