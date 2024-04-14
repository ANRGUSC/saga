# Parametric List Scheduler

## Running Experiments Locally

To get a local copy up and running on your machine follow these steps

### Installation

Clone the repository and cd into it:

```bash
git clone https://github.com/ANRGUSC/saga.git
cd saga
```

[Optional] Create a virtual environment and activate it
```bash
python -m venv saga_env
source saga_env/bin/activate # Unix or MacOS
saga_env\Scripts\activate # Windows
```
Switch branch to "feature/experimentation"

```bash
git checkout feature/experimentation
```
Install the required libraries
```bash
pip install .
```
### Running the experiments

Naviagte to "scripts/reproduce-parametric-sc.ipynb" and run all 


## Running Experiments on Coder

To run the experiments on our cloud servers without having to install anything locally


1. Go to https://coder.eclipse.usc.edu/

2. Enter Login credentials (Contact us to create a new account)

3. Click on create a workspace and select "saga-parametric"

4. Enter a name for your workspace and click on "Create Workspace"

5. Wait for it to setup and then click on "code-server"

6. Navigate to "scripts/reproduce-parametric-sc.ipynb" 

7. Click on "Select Kernel" and then install the recommended extensions

8. Once the extensions are installed select "Python Environments..." and then select "saga"

9. Click on "Run All"
