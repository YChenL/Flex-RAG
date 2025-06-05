from setuptools import setup, find_packages

setup(
    name="flexrag",
    version="0.1.0",
    description="A flexible framework for RAG engineering",
    author="Yangfu Li",
    author_email="yfli_cee@stu.ecnu.edu.cn",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        "magic-pdf[full]",
        "rank_bm25",
        "numpy==1.26.4",
        "scipy==1.11.4",
        "scikit-learn==1.4.2",
        "faiss-gpu",
        "transformers==4.44.2",
        "accelerate==0.17.0",
        "sentencepiece>=0.1.97",
        "huggingface-hub>=0.13.4",
        "tokenizers>=0.13.2",
        "safetensors>=0.3.2",
        "Pillow>=9.5.0",
        "sentence-transformers>=2.2.2",
        "langchain>=0.2.0",
        "langchain-openai>=0.0.1",
        "langchain-community>=0.0.1",
        "ipykernel>=6.25.0"
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
