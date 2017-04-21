# IRDM_project2

## Prerequisite
- pandas
- tensorflow 
- numpy
- pickle
- csv
- sklearn
- matplotlib.pyplot

## Introduction
This repository does not include the MSLR-WEB10K data required. Data can be downloaded from [here](https://www.microsoft.com/en-us/research/project/mslr/). To run the code, ensure the data files train.txt and test.txt are in the same directory.

## main.py 
Python code for logistic regression. Running this generate three .csv files (predicted labels, ground truth labels and query IDs). They are required for evaluations These would be saved in the same directory. 

## ndcg2.py
Python code that loads the .csv files and generate NDCG metric.

## map.py
Python code that loads the .csv files and generate MAP metric.

