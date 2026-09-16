"""Setuptools configuration for AgentLens."""

from setuptools import find_packages, setup

setup(
    name="agentlens",
    version="0.1.0",
    description="Local tracing and auditing for LangChain agents",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "anthropic",
        "python-dotenv",
        "langchain-groq",
    ],
    entry_points={"console_scripts": ["agentlens=agentlens.cli:main"]},
)
