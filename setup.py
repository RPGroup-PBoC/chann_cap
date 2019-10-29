import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ccutils",
    version="0.0.1",
    author="Manuel Razo, Rob Phillips",
    author_email="mrazomej {at} caltech {dot} edu",
    description="This repository contains all active research materials for a project attempting to calculate information processing capacity from a simple-repression motif.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrazomej/chann_cap.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
