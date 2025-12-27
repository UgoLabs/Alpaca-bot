from setuptools import setup, find_packages

setup(
    name="alpaca-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "alpaca-trade-api",
        "alpaca-py",
        "pandas",
        "numpy",
        "torch",
        "gym",
        "python-dotenv",
        "yfinance",
        "pytz",
        "transformers",
        "tokenizers",
    ],
)
