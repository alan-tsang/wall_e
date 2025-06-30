from setuptools import setup, find_packages

setup(
    name='wall_e',
    version='1.0.0',
    packages = find_packages(exclude=['test', 'test.*']),
    url='https://www.github.com/alan-tsang/wall_e',
    license='LGPL',
    author='Zhicun Zeng',
    author_email='zeiton@csu.edu.cn',
    description='some algorithms and a personal deep learning experiment framework for pytorch',
    install_requires=[
        "torch>=1.11.0",
        "datasets",
        "omegaconf>=1.4.0",
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
