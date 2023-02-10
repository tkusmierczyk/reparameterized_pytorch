import setuptools
from setuptools import setup

from reparameterized import __version__

setup(
    name="reparameterized",
    version=__version__,
    url="https://github.com/tkusmierczyk/reparameterized_pytorch",
    author="Tomasz Kuśmierczyk",
    author_email="tomasz.kusmierczyk@gmail.com",
    py_modules=["reparameterized"],
    packages=setuptools.find_packages(),
)
