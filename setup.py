from setuptools import setup, find_packages
from acm._version import __version__

setup(
    name = 'acm.py',
    packages = find_packages(),
    author = 'Dani Wiepert',
    python_requires='>=3.10',
    install_requires=[
        "numpy==2.4.6",
        "pandas==3.0.5",
        "torch==2.12.1",
        "torchaudio==2.11.0",
        "torchvision==0.27.1", 
        "torchcodec==0.14.0",
        "scikit-learn==1.9.0",
        "google-cloud-storage==3.12.0"
    ],
    include_package_data=False,
    version = __version__
)
