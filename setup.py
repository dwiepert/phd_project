from setuptools import setup, find_packages
from acm._version import __version__

setup(
    name = 'acm.py',
    packages = find_packages(),
    author = 'Dani Wiepert',
    python_requires='>=3.10',
    install_requires=[
    ],
    include_package_data=False,
    version = __version__
)
