# Parametric List Scheduler

<!-- table of contents -->
## Table of Contents
- [Preliminaries](#preliminaries)
- [Running Experiments Locally](#running-experiments-locally)
  - [Installation](#installation)
  - [Running the experiments](#running-the-experiments)
- [Running Experiments on Coder](#running-experiments-on-coder)

## Preliminaries
The general requirements are Python 3.10 (other versions may work but have not been tested).
We have tested the following instructions on Ubuntu 20.04 and MacOS 11.2.3.

## Running Experiments Locally

To get a local copy up and running on your machine follow these steps

### Installation

Clone the repository and cd into it:

```bash
git clone https://github.com/ANRGUSC/saga.git
cd saga
```

Create a virtual environment, activate it and set up the kernal
```bash
$conda create -n saga_env
$conda activate -n saga_env 
(saga_env)$conda install ipykernal
(saga_env)$ipython kernal install --user --name=saga_env
(saga_env)$pip install anrg.saga
```
Switch branch to "feature/experimentation"

```bash
(saga_env)$git checkout feature/experimentation
```
Install the required libraries
```bash
(saga_env)$pip install .
(saga_env)$conda deactivate
```
### Running the experiments

1. Activate your base conda environment or whichever environment has jupyter
2. Select "saga_env" as your kernal
3. Run all cells
```bash
(base)$jupyter notebook scripts/reproduce-parametric-sc.ipynb
```
Naviagte to "scripts/reproduce-parametric-sc.ipynb" and run all 


## Running Experiments on Coder

To run the experiments on our cloud servers without having to install anything locally


1. Go to https://coder.eclipse.usc.edu/

2. Enter Login credentials (Contact us to create a new account)

3. Click on create a workspace and select "saga-parametric"

4. Enter a name for your workspace and click on "Create Workspace"

5. Wait for the setup script to finish *completely* (the setup stdout window will close when it isi finished) and then click on "code-server"

6. Navigate to "scripts/reproduce-parametric-sc.ipynb" 

7. Click on "Select Kernel" and then install the recommended extensions

8. Once the extensions are installed select "Python Environments..." and then select "saga"

9. Click on "Run All". Make sure to leave the window open *the entire time* the experiments are running.
