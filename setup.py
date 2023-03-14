import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="myutils",
    version="0.0.1",
    author="pyb0924",
    author_email="pyb0924@sjtu.edu.cn",
    description="A library for my research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pyb0924/myutils",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
