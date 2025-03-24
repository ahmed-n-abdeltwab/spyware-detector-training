from setuptools import setup, find_packages

setup(
    name="spyware-detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas==1.5.3",
        "numpy==1.23.5",
        "scikit-learn==1.2.2",
        "PyYAML==6.0",
    ],
    python_requires=">=3.9",
)
