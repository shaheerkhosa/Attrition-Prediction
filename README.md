# Attrition-Prediction
A project that can predict the attrition of employees using various prediction models both with GUI and JSON/String input.

Dataset:
The dataset used in this project is th IBM HR Analytic dataset from Kaggle (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
The dataset contains various columns, both numerical and categorical.

Preprocessing:
The data is preprocessed by encoding the categorical columns, normalizing the numerical columns, and by smoothening that data to handle the class imbalance.

Preditiction:
This project allows you to select the model you wish to use to predict. It also shows handy evaluation criteria including ROC curve, confusion table, precision, and recall to illustrate the effectiveness of each model.

How to run:
After opening this folder in your code editor, type "streamlit run FrontEnd.py" in the terminal.
Alternatively, the same thing can be done by navigating to this directory in the command prompt and running the same command.