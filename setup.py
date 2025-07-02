from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="clustflow",
    version="0.1.0",
    description="Flexible clustering and embedding framework for mixed-type data",
    author="Your Name",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)