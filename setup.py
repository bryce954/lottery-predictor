from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lottery-predictor",
    version="1.0.0",
    author="Lottery Predictor",
    author_email="noreply@example.com",
    description="AI-powered Florida Lotto & Powerball number predictor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lottery-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dl": ["tensorflow>=2.13.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
    entry_points={
        "console_scripts": [
            "lottery-predictor=lottery_predictor.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "lottery_predictor": ["data/*.json"],
    },
)