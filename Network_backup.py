# %% import modules and def func
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from car_insurance import gridsearch_refit_plot


def cv_rmse_refit_on_whole(reg, X, y, kf, title, plot=True):
    """ no parameter tuning, just evaluate with 10fold cv and
        visualize fitting with whole dataset"""
    scores = cross_validate(reg, X, y, cv=kf, scoring='neg_mean_squared_error',
                            return_train_score=True)
    rmse_df = (-pd.DataFrame({'train RMSE': scores['train_score'],
                              'test RMSE': scores['test_score']})).applymap(np.sqrt)
    rmse_df.loc['average', 'train RMSE'] = np.mean(rmse_df['train RMSE'])
    rmse_df.loc['average', 'test RMSE'] = np.mean(rmse_df['test RMSE'])

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        y_pred = reg.fit(X, y).predict(X)
        residuals = y - y_pred
        ax1.plot(y, y_pred, '.')
        ax1.set_xlabel('observed y')
        ax1.set_ylabel('predicted y')
        ax1.set_title(title + ', predicted vs observed')

        ax2.plot(y_pred, residuals, '.')
        ax2.set_xlabel('predicted y')
        ax2.set_ylabel('residuals (y-y_pred')
        ax2.set_title(title + ', residual plot')
        plt.show()
    return rmse_df


def sweep_random_forest(n_list, d_list, m_list):
    results = []
    for m in m_list:
        for d in d_list:
            for n in n_list:
                if n % 20 == 1:
                    print('m={:}, n={:}'.format(m, n))
                reg = RandomForestRegressor(n_estimators=n, max_depth=d, bootstrap=True, max_features=m,
                                            oob_score=True, n_jobs=-1)
                reg.fit(X, y)
                oob_error = 1 - reg.oob_score_
                mean_test_rmse = cv_rmse_refit_on_whole(reg, X, y, kf, title='rfr', plot=False).loc[
                    'average', 'test RMSE']
                results.append(dict(n=n, d=d, m=m, oob_error=oob_error, mean_test_rmse=mean_test_rmse))
    results_df = pd.DataFrame(results)
    return results_df


# %% codes only run in main
def main():
    # %% load dataset
    Dataset = pd.read_csv('network_backup_dataset.csv')
    day_agg = Dataset.groupby(['Week #', 'Day of Week'],
                              as_index=False, sort=False)['Size of Backup (GB)'].sum()

    plt.plot(day_agg['Size of Backup (GB)'].iloc[:20])
    plt.xlabel('day number')
    plt.ylabel('Backup size (GB)')
    plt.xticks(np.arange(20))
    plt.show()

    plt.plot(day_agg['Size of Backup (GB)'].iloc[:105])
    plt.xlabel('day number')
    plt.ylabel('Backup size (GB)')
    plt.show()

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    le = LabelEncoder()
    cat_columns = ['Day of Week', 'Work-Flow-ID', 'File Name']
    y = Dataset['Size of Backup (GB)']

    # %% (a) linear regression, label encoding
    X = Dataset.drop(columns=['Backup Time (hour)', 'Size of Backup (GB)'])
    for col in cat_columns:
        X[col] = le.fit_transform(X[col])

    reg = LinearRegression()
    rmse_lr = cv_rmse_refit_on_whole(reg, X, y, kf, title='linear regression')

    # %% (b1) random forest regression with label encoding
    X = Dataset.drop(columns=['Backup Time (hour)', 'Size of Backup (GB)'])
    for col in cat_columns:
        X[col] = le.fit_transform(X[col])

    reg = RandomForestRegressor(n_estimators=20, max_depth=4, bootstrap=True, max_features=5, oob_score=True)
    rmse_rfr = cv_rmse_refit_on_whole(reg, X, y, kf, title='random forest regression')
    reg.fit(X, y)
    print('out of bag error: ', 1 - reg.oob_score_)

    # %% (b2) sweep for n_estimators, max_features
    n_list = np.arange(1, 122, 5)
    d_list = [4]
    m_list = np.arange(1, 6)
    results_df = sweep_random_forest(n_list, d_list, m_list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for m in m_list:
        results_m = results_df[results_df['m'] == m]
        ax1.plot(results_m['n'], results_m['mean_test_rmse'], label='max_feature={:}'.format(m))
        ax2.plot(results_m['n'], results_m['oob_error'], label='max_feature={:}'.format(m))
    ax1.set_title('mean test rmse')
    ax1.set_xlabel('number of trees')
    ax1.legend()
    ax2.set_title('oob error')
    ax2.set_xlabel('number of trees')
    ax2.legend()
    plt.show()

    # %% (b3) sweep max_depth from 1 to 10, with n=50, m=4
    n_list = [50]
    d_list = np.arange(1, 20, 2)
    m_list = [4]
    results_df = sweep_random_forest(n_list, d_list, m_list)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.plot(results_df['d'], results_df['mean_test_rmse'], label='n={:}, m={:}'.format(*n_list, *m_list))
    ax2.plot(results_df['d'], results_df['oob_error'], label='n={:}, m={:}'.format(*n_list, *m_list))
    ax1.set_title('mean test rmse')
    ax1.set_xticks(d_list)
    ax1.set_xlabel('max depth')
    ax1.legend()
    ax2.set_title('oob error')
    ax2.set_xticks(d_list)
    ax2.set_xlabel('max depth')
    ax2.legend()
    plt.show()

    # %% (b4) random forest feature importance
    X = Dataset.drop(columns=['Backup Time (hour)', 'Size of Backup (GB)'])
    for col in cat_columns:
        X[col] = le.fit_transform(X[col])

    n, d, m = 50, 9, 4
    reg = RandomForestRegressor(n_estimators=n, max_depth=d, bootstrap=True, max_features=m,
                                oob_score=True, n_jobs=-1)
    reg.fit(X, y)
    importance = reg.feature_importances_
    plt.barh(X.columns.values, importance, height=0.5, label='m={:}, d={:}, n={:}'.format(m, d, n))
    plt.ylabel('importance')
    plt.legend()
    plt.show()
    rmse_rfr = cv_rmse_refit_on_whole(reg, X, y, kf, title='random forest regression')

    # %% (b5) visualize one tree from random forest
    n, d, m = 50, 4, 4
    reg = RandomForestRegressor(n_estimators=n, max_depth=d, bootstrap=True, max_features=m,
                                oob_score=True, n_jobs=-1)
    reg.fit(X, y)
    tree = reg.estimators_[0]
    export_graphviz(tree, out_file='tree.dot',
                    feature_names=X.columns.values,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    (graph,) = pydot.graph_from_dot_file('tree.dot')
    graph.write_png('tree.png')

    # %% (c1) neural network regression, sweep hidden layer size and activation function
    X = Dataset.drop(columns=['Backup Time (hour)', 'Size of Backup (GB)'])
    X = pd.get_dummies(X, columns=cat_columns)
    size_list = [2, 5, 10, 50, 100, 150, 200, 300, 400, 500, 600]
    activation_list = ['relu', 'logistic', 'tanh']

    for activation in activation_list:
        results_nn = []
        for size in size_list:
            reg = MLPRegressor(hidden_layer_sizes=(size,), activation=activation)
            rmse_nn = cv_rmse_refit_on_whole(reg, X, y, kf, title='NN regression', plot=False)
            results_nn.append(dict(activation=activation, size=size, rmse=rmse_nn.loc['average', 'test RMSE']))
        results_nn_df = pd.DataFrame(results_nn)
        plt.plot(results_nn_df['size'], results_nn_df['rmse'], label=activation)
        plt.xlabel('number of hidden units')
    plt.legend()
    plt.show()

    # %% (c2) NN regression, fit on whole with best parameters and plot
    X = Dataset.drop(columns=['Backup Time (hour)', 'Size of Backup (GB)'])
    X = pd.get_dummies(X, columns=cat_columns)
    reg = MLPRegressor(hidden_layer_sizes=(400,), activation='relu')
    rmse_nn = cv_rmse_refit_on_whole(reg, X, y, 2, title='NN regression', plot=True)

    # %% (d1) Predict the Backup size for each of the workflows separately
    X = Dataset[Dataset['Work-Flow-ID'] == 'work_flow_0'].drop(
        columns=['Backup Time (hour)', 'Work-Flow-ID', 'Size of Backup (GB)'])
    y = Dataset[Dataset['Work-Flow-ID'] == 'work_flow_0']['Size of Backup (GB)']
    for col in ['Day of Week', 'File Name']:
        X[col] = le.fit_transform(X[col])

    reg = LinearRegression()
    rmse_lr_0 = cv_rmse_refit_on_whole(reg, X, y, kf, title='linear regression, flow 0', plot=True)

    # %% (d2) polynomial regression, sweep degree of polynomial
    rmse_poly = []
    for k in range(1, 14, 2):
        poly = PolynomialFeatures(degree=k, interaction_only=False)
        X_ = poly.fit_transform(X)
        rmse_poly_k = cv_rmse_refit_on_whole(reg, X_, y, kf, title='poly regression, flow 0', plot=False)
        rmse_poly.append(dict(k=k, train_rmse=rmse_poly_k.loc['average', 'train RMSE'],
                              test_rmse=rmse_poly_k.loc['average', 'test RMSE']))
    rmse_poly_df = pd.DataFrame(rmse_poly)
    plt.plot(rmse_poly_df['k'], rmse_poly_df['train_rmse'], label='train rmse')
    plt.plot(rmse_poly_df['k'], rmse_poly_df['test_rmse'], label='test rmse')
    plt.xlabel('degree of polynomial')
    plt.legend()
    plt.show()

    # %% (e1) KNN regression, sweep k
    y = Dataset['Size of Backup (GB)']
    X = Dataset.drop(columns=['Backup Time (hour)', 'Size of Backup (GB)'])
    for col in cat_columns:
        X[col] = le.fit_transform(X[col])

    reg = KNeighborsRegressor(n_neighbors=4)
    param_grid = {'n_neighbors': np.arange(1, 11)}
    gridsearch_refit_plot(X, y, reg, param_grid, kf, 'KNN regression')


if __name__ == '__main__':
    main()
