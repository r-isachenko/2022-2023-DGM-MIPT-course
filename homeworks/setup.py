from distutils.core import setup
from setuptools import find_packages

setup(
    name="dgm_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
        "tqdm",
        "torch",
        "torchvision",
        "torchaudio",
        "gdown",
        "scikit-learn",
    ],
)
