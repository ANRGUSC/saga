from setuptools import setup, find_packages

setup(
    name="anrg-saga",
    version="1.0.0",
    author="ANRG USC", 
    author_email="anrg@usc.edu",
    description="Scheduling Algorithms Gathered - collection of task graph scheduling algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ANRGUSC/saga",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "networkx",
        "numpy", 
        "pandas",
        "scipy",
        "matplotlib",
        "joblib>=1.3.2",
        
        # Analysis and stats
        "statsmodels>=0.14.1",
        "seaborn>=0.13.2",
        
        # Plotting
        "plotly",
        "kaleido",

        "gitpython",
        
        # Workflow support
        "wfcommons",
        
        # Utilities
        "dill",
    ],
    extras_require={
        "smt": ["pysmt"],
        "viz": ["pygraphviz"], 
        "web": ["streamlit"],
        "test": ["pytest>=7.0.0", "pytest-timeout>=2.1.0"],
        "all": [
            "pysmt", 
            "pygraphviz", 
            "streamlit",
            "pytest>=7.0.0", 
            "pytest-timeout>=2.1.0"
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
    ],
)