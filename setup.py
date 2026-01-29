from setuptools import setup, find_packages
from acm._version import __version__

setup(
    name = 'acm.py',
    packages = find_packages(),
    author = 'Dani Wiepert',
    python_requires='>=3.10',
    install_requires=[
        "numpy==2.2.6",
        "pandas==2.3.3",
        "torch==2.10.0",
        "torchaudio==2.10.0",
        "torchvision==0.25.0"
    ],
    include_package_data=False,
    version = __version__
)
