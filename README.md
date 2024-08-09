
Python Assessment tool:
The submission zip file with my code.
Open Anaconda Navigator, and select Jupyter Notebook. Launch ‘SDSS_Data_Analysis_and_Machine_Learning_Model.ipynb’ from your directory/download. Make sure you have the required libraries installed, including:
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from collections import Counter

Other libraries are implemented in the code in case I missed any.

Select ‘Cell‘, ‘Run All‘ to run the file. The model is documented in comments throughout the
file. Change the Excel file with your data to change the source.
The next step is to load the csv file into pandas data frame using pandas.read_csv. The csv file used to run the code can be found in the resources folder.
The pdf version of the code and all the associated elements have also been uploaded. 

Models:
I have used two models to compare the performance and have written different scenarios to study the impact of data cleaning, model class distribution, and dimensionality reduction on the accuracy, recall value, and f1 values of the models
1. RandomForestClassifier: from sklearn.ensemble import RandomForestClassifier
2. Support Vector Classifier: from sklearn.svm import SVC

Different model accuracies are stored to check the final performance and we run a grid search to find the best hyperparameter.
The final Model is run using the parameters derived by grid search.
