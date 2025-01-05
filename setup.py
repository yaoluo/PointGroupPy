from setuptools import setup, find_packages

setup(
    name="PointGroupPy",
    version="0.1.0",
    author="Yao Luo",
    author_email="yluo7@caltech.edu",
    description="A library for finite matrix group, typically for point group.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yaoluo/PointGroupPy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
