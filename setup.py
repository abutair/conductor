from setuptools import setup, find_packages

setup(
    name="conductor",
    version="0.1.0",
    description="Orchestrating language models with intuitive reinforcement learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "tqdm>=4.62.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)