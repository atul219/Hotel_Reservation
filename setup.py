from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name= "Hotel-Reservations-MLOPS-1",
    version = "0.1",
    author = "Atul",
    packages= find_packages(),
    install_requires = requirements,
)