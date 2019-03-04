#%% import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from Network_backup import cv_rmse_refit_on_whole
import statsmodels.api as sm

#%% write codes in main
if __name__ == '__main__':
    # %% load dataset
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
               'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    Dataset = pd.read_csv('housing_data.csv', header=None)
    Dataset.columns = columns
    X = Dataset.drop(columns=['MEDV'])
    y = Dataset['MEDV']

    #%% 2a use statsmodel to get variable statistics
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    #%% 2b cv_rmse_refit_on_whole
    reg = LinearRegression()
    kf = KFold(10, shuffle=True, random_state=42)
    rmse_lr = cv_rmse_refit_on_whole(reg, X, y, kf, 'linear regression', plot=True)
    reg.fit(X, y)
    coef = reg.coef_

    #%% 3 regularization
    lasso = Lasso(alpha)
    ridge = Ridge(alpha)
    en = ElasticNet(alpha, l1_ratio)