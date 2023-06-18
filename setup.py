from setuptools import setup

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines()]

setup(
    name="density_decoding",
    version="0.1",
    packages=["density_decoding"],
    install_requires=require,
)
