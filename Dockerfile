# Use an official Python image as the base image
FROM python:3.10-slim

# Install system dependencies required for pysmt, wfcommons, and other solvers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-dev \
    libgsl-dev \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./src /app

# Install package dependencies and pytest-related tools
RUN pip install -e . \
    && pip install pytest pytest-timeout

# Install Z3 solver
RUN pysmt-install --z3 --confirm-agreement

COPY ./tests /app/tests

# Run the tests on container startup
CMD ["pytest", "./tests", "--timeout=60"]
