# packages for data analysis & mapping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from math import sqrt
import seaborn as sns
from datetime import date 

# modeling methods
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE 
import sklearn.preprocessing

# address warnings
import warnings
warnings.filterwarnings("ignore")

import os.path

def get_lemonade_data(cached=False):
    '''
    This function reads in college data from a URL and writes data to
    a csv file if cached == False or if cached == True reads in college df from
    a csv file, returns df.
    '''
    url = 'https://gist.githubusercontent.com/ryanorsinger/19bc7eccd6279661bd13307026628ace/raw/e4b5d6787015a4782f96cad6d1d62a8bdbac54c7/lemonade.csv'
    
    if cached == False or os.path.isfile('lemonade.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = pd.read_csv(url)
        
        # Write DataFrame to a csv file.
        df.to_csv('lemonade.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('lemonade.csv', index_col=0)
        
    return df