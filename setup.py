from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mvkm-ed",
    version="1.1.0",
    author="Kristina P. Sinaga",
    author_email="kristinasinaga41@gmail.com",
    description="Federated Multi-View K-Means Clustering with Rectified Gaussian Kernel",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Fed-MVKM",
    keywords="clustering, federated-learning, multi-view-clustering, k-means, privacy-preserving, machine-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
    ],
)
