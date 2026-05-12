"""AniRec v22 — Multi-Domain Anime & Movie Recommender."""

from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent
long_description = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="anirec",
    version="22.0.0",
    description="Multi-domain anime and movie recommendation system (LightGCN + SASRec + NCF)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AniRec Contributors",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "sentence-transformers>=2.2.0",
        "pandas>=2.0.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "nest_asyncio>=1.5.6",
        "thefuzz>=0.19.0",
        "python-Levenshtein>=0.20.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.6.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "anirec-train=train:main",
            "anirec-infer=infer:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)