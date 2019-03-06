#%% import modules
""" (1) avoid circular import
    (2) create __init__.py file to make the dir as a module
    (3) from codes.file import func
    (4) if marking codes as 'source code', no need to type codes.xx during
    import, it appends the codes dir into sys.path
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from codes.Network_backup import gridsearch_refit_plot
from codes.Network_backup import cv_rmse_refit_on_whole
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression


def cv_rmse_log(reg, X, y, kf):
    MSE = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg.fit(X_train, np.log(y_train))
        logy_pred = reg.predict(X_test)
        MSE.append(np.mean((np.exp(logy_pred) - y_test) ** 2))
    return np.mean(np.sqrt(MSE))


#%% write codes in main
def main():
    # %% load dataset
    Dataset = pd.read_csv('dataset/insurance_data.csv')
    X = Dataset.drop(columns=['charges'])
    y = Dataset['charges']
    kf = KFold(10, shuffle=True, random_state=42)
    reg = LinearRegression()

    #%% (1a) one-hot encoding
    cat_columns = ['ft4', 'ft5', 'ft6']
    X_onehot = pd.get_dummies(X, columns=cat_columns)
    rmse_onehot = cv_rmse_refit_on_whole(reg, X_onehot, y, kf,
                                         title='one-hot encoding', plot=True)

    #%% (1b) standardize num features
    num_columns = ['ft1', 'ft2', 'ft3']
    X_standard = X_onehot.copy()
    X_standard[num_columns] = StandardScaler().fit_transform(X_standard[num_columns])
    rmse_standard = cv_rmse_refit_on_whole(reg, X_standard, y, kf,
                                           title='standardization', plot=True)

    #%% (1c) cut ft1, standardize ft2-3, one-hot ft4-6
    X_c = X_onehot.copy()
    X_c[['ft2', 'ft3']] = StandardScaler().fit_transform(X_c[['ft2', 'ft3']])
    X_c['ft1'] = pd.cut(X_c['ft1'], bins=[X_c['ft1'].min()-1, 30, 50, X_c['ft1'].max()],
                        labels=[1, 2, 3])
    rmse_c = cv_rmse_refit_on_whole(reg, X_c, y, kf,
                                    title='three transformations', plot=True)

    #%% (2) corr exploration
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

    #%% (3a) log transform y
    rmse_log = cv_rmse_log(reg, X_c, y, kf)

    # %% (3b) corr exploration with log(y)
    f, pval = f_regression(X_le, np.log(y))
    mi = mutual_info_regression(X_le, np.log(y))
    corr_df = pd.DataFrame([f, pval, mi], columns=X_le.columns.values,
                           index=['F value', 'p value', 'MI']).T

    plt.scatter(X_le['ft2'], np.log(y), c=X_le['ft5'])
    plt.xlabel('ft2')
    plt.ylabel('log(charges)')
    plt.title('log(charges) vs ft2')
    plt.show()

    plt.scatter(X_le['ft1'], np.log(y), c=X_le['ft5'])
    plt.xlabel('ft1')
    plt.ylabel('log(charges)')
    plt.title('log(charges) vs ft1')
    plt.show()

    # %% (4a) polynomial features, interaction only
    rmse_poly = []
    for k in range(1, 6):
        poly = PolynomialFeatures(degree=k, interaction_only=True)
        X_ = poly.fit_transform(X_le)
        rmse_poly_k = cv_rmse_refit_on_whole(reg, X_, y, kf,
                                             title='poly regression',
                                             plot=False)
        rmse_poly.append(
            dict(k=k, train_rmse=rmse_poly_k.loc['average', 'train RMSE'],
                 test_rmse=rmse_poly_k.loc['average', 'test RMSE']))
    rmse_poly_df = pd.DataFrame(rmse_poly)
    plt.plot(rmse_poly_df['k'], rmse_poly_df['train_rmse'], label='train rmse')
    plt.plot(rmse_poly_df['k'], rmse_poly_df['test_rmse'], label='test rmse')
    plt.xlabel('degree of polynomial')
    plt.title('interaction only poly win X_le')
    plt.legend()
    plt.show()

    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_poly = poly.fit_transform(X_le)
    rmse_poly = cv_rmse_refit_on_whole(reg, X_poly, y, kf,
                                       title='poly regression, 2',
                                       plot=True)

    #%% (4b) more models: NN regression
    X_le = X.copy()
    X_le[cat_columns] = X_le[cat_columns].apply(LabelEncoder().fit_transform)

    param_grid = {'hidden_layer_sizes': [(50,), (100,), (200,), (300,)]}
    reg = MLPRegressor(activation='relu', learning_rate_init=0.06, max_iter=800)
    gridsearch_refit_plot(X_le, y, reg, param_grid, kf, 'NN regression')

    # %% (4b) more models: random forest regression
    param_grid = {'n_estimators': [2, 5, 10, 20, 50, 100]}
    reg = RandomForestRegressor(max_depth=6, n_jobs=-1)
    gridsearch_refit_plot(X_le, y, reg, param_grid, kf, 'Random forest')

    # %% (4b) more models: gradient boosting regression
    param_grid = {'n_estimators': [2, 10, 20, 50, 100, 200, 400]}
    reg = GradientBoostingRegressor()
    gridsearch_refit_plot(X_le, y, reg, param_grid, kf, 'Gradient boosting')


if __name__ == '__main__':
    main()
