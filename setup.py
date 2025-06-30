from setuptools import setup, find_packages

setup(
    name='wall_e',
    version='1.0.0',
    packages = find_packages(),
    url='https://www.github.com/alan-tsang/wall_e',
    license='LGPL',
    author='Zhicun Zeng',
    author_email='zeiton@csu.edu.cn',
    description='some algorithms and a personal deep learning experiment framework for pytorch',
    install_requires=[
        "torch>=1.11.0", # 支持torchrun的版本
        "datasets",
        "omegaconf>=1.4.0", # 之前是否支持from_cli没有查到
        "numpy",
        "pandas",
        "pyyaml",
        "ray",
        "transformers",
        "rdkit",
        "wandb"
    ],
    extras_require = {
        "deepspeed": ["deepspeed"],
        "gnn": ["torch_geometric", "torch_scatter"],
        "other": ["matplotlib", "torchviz", "graphviz", "pynvml"]
    },
)
