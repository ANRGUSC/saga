from setuptools import setup, find_packages
import pathlib

thisdir = pathlib.Path(__file__).parent.absolute()

long_description = (thisdir / "README.md").read_text()

setup(
    name='anrg.saga',
    version='0.0.6',
    description='Collection of schedulers for distributed computing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ANRGUSC/saga',
    author='Jared Coleman',
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        'networkx',
        'numpy',
        'pygraphviz',
        'matplotlib',
        'scipy',
        'pandas',
        'plotly',
    ],
    python_requires='>=3.6',
)
