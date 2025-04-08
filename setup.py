from setuptools import setup, find_packages

# Read production requirements
with open("requirements-prod.txt", "r") as f:
    prod_requirements = [line.strip() for line in f.readlines() if line.strip()]

# Read development requirements
with open("requirements-dev.txt", "r") as f:
    dev_requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="poe_api_bridge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=prod_requirements,
    extras_require={
        "dev": dev_requirements,
    },
    python_requires=">=3.8",
)
