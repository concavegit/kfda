import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kfda",
    version="0.1.0",
    author="Kawin Nikomborirak",
    author_email="concavemail@gmail.com",
    description="Kernel FDA implementation described in https://arxiv.org/abs/1906.09436",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/concavegit/kfda",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
