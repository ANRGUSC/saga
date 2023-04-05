from setuptools import setup, find_packages

setup(
    name='anrg.saga',
    version='0.0.1',
    description='Collection of schedulers for distributed computing',
    author='Jared Coleman',
    packages=['saga'],
    install_requires=[
        'networkx',
        'numpy',
        'pygraphviz',
        'matplotlib'
    ],
    python_requires='>=3.6',
)
