import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

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
)
