from setuptools import setup, find_packages

setup(
    name="ft-platform",
    version="1.0.0",
    description="Self-service fine-tuning CLI for LLMs (MKD Co., Ltd.)",
    author="MKD AI Platform Team",
    author_email="platform@mkd.kr",
    python_requires=">=3.10",
    packages=find_packages(where="scripts"),
    package_dir={"": "scripts"},
    entry_points={
        "console_scripts": [
            "ft=ft:main",
        ],
    },
    install_requires=[
        "huggingface_hub>=0.23",
        "datasets>=2.19",
        "transformers>=4.41",
        "accelerate>=0.30",
        "peft>=0.11",
        "trl>=0.8",
        "bitsandbytes>=0.43",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25",
        "pandas>=2.2",
        "pyarrow>=16.0",
        "datasketch>=1.6",
        "requests>=2.31",
        "pyyaml>=6.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.111",
            "uvicorn[standard]>=0.30",
            "pydantic>=2.7",
            "redis>=5.0",
        ],
        "train": ["axolotl", "flash-attn"],
        "quant": ["autoawq"],
        "eval": ["lm_eval>=0.4"],
        "dev": [
            "pytest>=8.2",
            "pytest-cov>=5.0",
            "pytest-asyncio>=0.23",
            "httpx>=0.27",
            "black>=24.4",
            "ruff>=0.4",
            "mypy>=1.10",
            "bandit>=1.7",
        ],
    },
)
