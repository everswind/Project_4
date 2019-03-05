#%% import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from codes.Network_backup import cv_rmse_refit_on_whole
import statsmodels.api as sm


#%% write codes in main
def main():
    # %% load dataset
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
               'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    Dataset = pd.read_csv('dataset/housing_data.csv', header=None)
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

    #%% 3 L1 Lasso regularization
    param_grid = {'alpha': np.linspace(0.0001, 1, 20)}
    grid = GridSearchCV(Lasso(max_iter=3000), param_grid, cv=kf, scoring='neg_mean_squared_error',
                        return_train_score=True)
    grid.fit(X, y)
    cv_results = pd.DataFrame(grid.cv_results_)
    plt.plot(cv_results['param_alpha'], np.sqrt(-cv_results['mean_train_score']),
             label='train rmse')
    plt.plot(cv_results['param_alpha'], np.sqrt(-cv_results['mean_test_score']),
             label='test rmse')
    plt.legend()
    plt.xlabel('alpha')
    plt.title('Lasso regression')
    plt.show()

    #%% 3 L2 Ridge regularization
    param_grid = {'alpha': np.linspace(0.0001, 1, 20)}
    grid = GridSearchCV(Ridge(max_iter=3000), param_grid, cv=kf, scoring='neg_mean_squared_error',
                        return_train_score=True)
    grid.fit(X, y)
    cv_results = pd.DataFrame(grid.cv_results_)
    plt.plot(cv_results['param_alpha'], np.sqrt(-cv_results['mean_train_score']),
             label='train rmse')
    plt.plot(cv_results['param_alpha'], np.sqrt(-cv_results['mean_test_score']),
             label='test rmse')
    plt.legend()
    plt.xlabel('alpha')
    plt.title('Ridge regression')
    plt.show()

    #%% 3 ElasticNet
    alpha_list = np.linspace(0.0001, 1, 20)
    l1_list = np.linspace(0.01, 1, 4)
    param_grid = {'alpha': alpha_list, 'l1_ratio': l1_list}
    grid = GridSearchCV(ElasticNet(max_iter=5000), param_grid, cv=kf, scoring='neg_mean_squared_error',
                        return_train_score=True)
    grid.fit(X, y)
    cv_results = pd.DataFrame(grid.cv_results_)
    for l1 in l1_list:
        cv_l1 = cv_results[cv_results['param_l1_ratio']==l1]
        plt.plot(cv_l1['param_alpha'], np.sqrt(-cv_l1['mean_train_score']),
                 label='train rmse, l1_ratio={:}'.format(l1))
        plt.plot(cv_l1['param_alpha'], np.sqrt(-cv_l1['mean_test_score']),
                 label='test rmse, l1_ratio={:}'.format(l1))
    plt.legend()
    plt.xlabel('alpha')
    plt.title('ElasticNet regression')
    plt.show()

    #%% compare coef of variables
    coef = []

    linear = LinearRegression()
    linear.fit(X, y)
    coef.append(linear.coef_)

    lasso = Lasso(alpha=0.0001)
    lasso.fit(X, y)
    coef.append(lasso.coef_)

    ridge = Ridge(alpha=0.0001)
    ridge.fit(X, y)
    coef.append(ridge.coef_)

    elastic = ElasticNet(alpha=0.0001, l1_ratio=1)
    elastic.fit(X, y)
    coef.append(elastic.coef_)

    coef_T = np.transpose(coef)
    coef_df = pd.DataFrame(coef_T,
                           index=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                                    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
                           columns=['Linear', 'Lasso', 'Ridge', 'ElasticNet'])


if __name__ == '__main__':
    main()
