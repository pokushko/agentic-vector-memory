from setuptools import setup, find_packages

setup(
    name="agentic-vector-memory",
    version="1.0.0",
    description="Episodic and Semantic Dual-Store Memory Engine for Autonomous AI Agents",
    author="Roman Pokushko",
    author_email="pokusko.roman@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
