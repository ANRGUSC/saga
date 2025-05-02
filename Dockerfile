# Use an official Python image as the base image
FROM python:3.10-slim

# Install system dependencies required for pysmt, wfcommons, and other solvers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-dev \
    libgsl-dev \
    curl \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    python3-dev \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    g++ \
    ca-certificates \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./src /app

# Install package dependencies and pytest-related tools
RUN pip install -e .
RUN pip install pytest pytest-timeout
RUN pip install openai python-dotenv pymongo
RUN pip install pygraphviz

# Install Z3 solver
RUN pysmt-install --z3 --confirm-agreement
COPY ./tests /app/tests

# Set default command to bash (so you can interact with the container)
CMD ["/bin/bash"]
