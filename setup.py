import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
requirementPath = "./requirements.txt"
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="condolence_models",
    version="0.0.1",
    author="Naitian Zhou",
    author_email="naitian@umich.edu",
    description="Detecting condolence, distress, and empathy in text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=install_requires,
)
