from setuptools import setup, find_packages


# Function to read the contents of the requirements.txt file
def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# Load requirements
install_requires = load_requirements()

setup(
    name="image-network-streaming",
    version="0.1.0",
    author="Timo Leitritz",
    author_email="42964574+TimoIllusion@users.noreply.github.com",
    description="A test for image transmission via multiple communication systems: FastAPI, ZeroMQ (ZMQ), ImageZMQ, grpc.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TimoIllusion/image-network-streaming",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require={
        "test": ["pytest>=7.0.0, <8.0.0", "pytest-asyncio", "httpx"],
    },
)
