# Kidney-Disease-Classification-MLflow-DVC 


##Workflows 

1. Update config.yaml
2. Update secrets.yaml [optional]
3. Update params.yaml
4. Update the entity 
5. Update the configuration manager in src config 
6. Update the components
7. Update the pipeline 
8. Update the main.py 
9. Update the dvc.yaml 
10. app.py

# How to run?

### STEPS:
Clone the repository 

'''bash 

https://github.com/Anshika102005/Kidney-Disease-Classification-DL-Project-DVC

### STEP 01- Create a conda environment after opening the repository 

'''bash 
conda create -n cnncls python=3.8 -y
'''

'''bash 
conda activate cnncls
'''

### STEP 02- install the requirements 

'''bash 
pip install -r requirements.txt 
'''

### cmd
 -mlflow ui


### dagshub
[dagshub](https://dagshub.com/)

import dagshub
dagshub.init(repo_owner='Anshika102005', repo_name='Kidney-Disease-Classification-DL-Project-DVC', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)



Run this to export as env variables:

'''bash

export MLFLOW_TRACKING_URI = "https://dagshub.com/Anshika102005/Kidney-Disease-Classification-DL-Project-DVC.mlflow"

export MLFLOW_TRACKING_USERNAME = Anshika102005

export MLFLOW_TRACKING_PASSWORD = Anshika@102005

