from setuptools import setup, find_packages

setup(
    name="minigpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch", "tokenizers", "requests"],
    author="Your Name",
    author_email="your@email.com",
    description="A lightweight GPT-style transformer for research and inference.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/minigpt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
)