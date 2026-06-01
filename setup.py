from setuptools import setup, find_packages

setup(
    name="ft-platform",
    version="1.0.0",
    description="Self-service fine-tuning CLI for LLMs (MKD Co., Ltd.)",
    author="Hika",
    author_email="hikabarkif143@gmail.com",
    python_requires=">=3.10",
    packages=find_packages(where="scripts"),
    package_dir={"": "scripts"},
    entry_points={
        "console_scripts": [
            "ft=ft:main",
        ],
    },
    install_requires=[
        "huggingface_hub",
        "datasets",
        "transformers",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "pandas",
        "pyarrow",
        "datasketch",
        "requests",
        "pyyaml",
    ],
    extras_require={
        "train": ["axolotl", "flash-attn"],
        "quant": ["autoawq"],
        "eval": ["lm_eval"],
    },
)
