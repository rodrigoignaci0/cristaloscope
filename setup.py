from setuptools import setup, find_packages

setup(
    name="cristaloscope",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "transformers>=4.35",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "umap-learn",
    ],
    author="NE-OS Research",
    description="LLM internal space visualizer and hallucination interceptor",
    python_requires=">=3.9",
)
