from setuptools import setup

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines()]

setup(
    name="clusterless",
    version="0.1",
    packages=["clusterless"],
    install_requires=require,
)
