import setuptools

# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

# This call to setuptools.setup() does all the work
setuptools.setup(
    name="kidney-disease-classification",
    version="0.1.0",
    author="Prathamesh Bidkar",
    author_email="prathameshb2299@gmail.com",
    description="A simple kidney disease classification project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StealthedDefenderMe/kidney-disease-classification.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)