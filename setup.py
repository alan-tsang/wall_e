from setuptools import setup, find_packages

setup(
    name='wall_e',
    version='1.0.0',
    packages = find_packages(exclude=['test', 'test.*', 'tmp', 'model']),
    url='https://www.github.com/alan-tsang/wall_e',
    license='LGPL',
    author='Zhicun Zeng',
    author_email='zengzhicun@csu.edu.cn',
    description='a deep learning experiment framework based on  pytorch',
    install_requires=[
        "torch",
        "datasets",
        "omegaconf>=1.4.0",
        "numpy",
        "pandas",
        "pyyaml",
        "ray",
        "transformers",
        "rdkit",
        "wandb",
        "matplotlib"
    ],
    extras_require = {
        "deepspeed": ["deepspeed"],
        "gnn": ["torch_geometric", "torch_scatter"],
        "other": ["matplotlib", "torchviz", "graphviz", "pynvml"]
    },
)
