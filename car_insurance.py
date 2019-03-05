#%% import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from Network_backup import cv_rmse_refit_on_whole
import statsmodels.api as sm
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression

#%% write codes in main
if __name__ == '__main__':
    # %% load dataset
    Dataset = pd.read_csv('insurance_data.csv')
    X = Dataset.drop(columns=['charges'])
    y = Dataset['charges']
    kf = KFold(10, shuffle=True, random_state=42)
    reg = LinearRegression()

    #%% (1a) one-hot encoding
    cat_columns = ['ft4', 'ft5', 'ft6']
    X_onehot = pd.get_dummies(X, columns=cat_columns)
    rmse_onehot = cv_rmse_refit_on_whole(reg, X_onehot, y, kf, title='one-hot encoding', plot=True)

    #%% (1b) standardize num features
    num_columns = ['ft1', 'ft2', 'ft3']
    X_standard = X_onehot.copy()
    X_standard[num_columns] = StandardScaler().fit_transform(X_standard[num_columns])
    rmse_standard = cv_rmse_refit_on_whole(reg, X_standard, y, kf, title='standardization', plot=True)

    #%% (1c) cut ft1, standardize ft2-3, one-hot ft4-6
    X_c = X_onehot.copy()
    X_c[['ft2', 'ft3']] = StandardScaler().fit_transform(X_c[['ft2', 'ft3']])
    X_c['ft1'] = pd.cut(X_c['ft1'], bins=[X_c['ft1'].min()-1, 30, 50, X_c['ft1'].max()],
                        labels=[1, 2, 3])
    rmse_c = cv_rmse_refit_on_whole(reg, X_c, y, kf, title='three transformations', plot=True)

    #%% (2)
    X_le = X.copy()
    X_le[cat_columns] = X_le[cat_columns].apply(LabelEncoder().fit_transform)
    f, pval = f_regression(X_le, y)
    mi = mutual_info_regression(X_le, y)
    corr_df = pd.DataFrame([f, pval, mi], columns=X_le.columns.values,
                                index=['F value', 'p value', 'MI']).T

    plt.scatter(X_le['ft2'], y, c=X_le['ft5'])
    plt.xlabel('ft2')
    plt.ylabel('charges')
    plt.title('charges vs ft2')
    plt.show()

    plt.scatter(X_le['ft1'], y, c=X_le['ft5'])
    plt.xlabel('ft1')
    plt.ylabel('charges')
    plt.title('charges vs ft1')
    plt.show()