import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

"""
Fairness measurement: Mean Absolute Error (MAE) differences， statistical parity differences
"""


def SPD(y_pred, slabel_list, unpriv_group):
    sp_list = {}
    for g in set(slabel_list):
        y_pred_g = y_pred[slabel_list == g]
        sp_g = np.mean(y_pred_g)
        sp_list.setdefault(g, sp_g)

    max_sp = -np.inf
    for g in sp_list:
        if (g != unpriv_group):
            sp = sp_list[g] - sp_list[unpriv_group]
            if (sp > max_sp):
                max_sp = sp
                max_g = g
    return max_g, max_sp


def EAD(y_gt, y_pred, slabel_list, unpriv_group):
    mae_list = {}
    for g in set(slabel_list):
        y_truth_g = y_gt[slabel_list == g]
        y_pred_g = y_pred[slabel_list == g]
        mae_g = mean_absolute_error(y_truth_g, y_pred_g)
        mae_list.setdefault(g, mae_g)

    max_ead = -np.inf
    for g in mae_list:
        if (g != unpriv_group):
            mae = mae_list[g] - mae_list[unpriv_group]
            if (mae > max_ead):
                max_ead = mae
                max_g = g
    return max_g, max_ead


def model(df_train, y_train, df_test, y_test, selected_features, label, slabel, model='lr'):
    # training
    df_f = df_train[[slabel] + selected_features].copy()
    df_f[label] = y_train
    df_f = df_f.dropna()
    X = df_f[selected_features]
    y = df_f[label]

    # kf=KFold(n_splits=10,random_state=0)
    if (model == 'lr'):
        regr = LinearRegression()
    elif (model == 'log'):
        regr = LogisticRegression()
    elif (model == 'rf'):
        regr = RandomForestRegressor(max_depth=5)
    elif (model == 'tree'):
        regr = DecisionTreeRegressor(max_depth=2)
    regr.fit(X, y)

    ##prediction
    X_test = df_test[selected_features + [slabel]]

    y_pred = regr.predict(X_test[selected_features])

    return y_pred, y_test
