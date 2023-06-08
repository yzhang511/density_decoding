import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score

seed = 666
np.random.seed(seed)



def discrete_decoder(
    x, 
    y, 
    train, 
    test, 
    penalty_type='l2', 
    penalty_strength=1000, 
    verbose=True
):
    '''
    Inputs:
    ------
    x: (k,c,t) or (k,t,c) array.
    y: (k,) array.
    train: training trial indices.
    test: test trial indices. 
    '''
    x_train = x.reshape(-1, x.shape[1] * x.shape[2])[train]
    x_test = x.reshape(-1, x.shape[1] * x.shape[2])[test]
    y_train = y[train]
    y_test = y[test]
    lr = LogisticRegression(random_state=seed, 
                            max_iter=10000, 
                            tol = 0.01, 
                            solver='liblinear',
                            penalty=penalty_type, 
                            C=penalty_strength)
    lr.fit(x_train, y_train)
    y_prob = lr.predict_proba(x_test)
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        print(f'accuracy: {acc:.3f}')
    return y_train, y_test, y_pred, y_prob, acc


def continuous_decoder(
    x, 
    y, 
    train, 
    test,
    penalty_strength=1000,
    verbose=False
):
    '''
    Inputs:
    ------
    x: (k,c,t) or (k,t,c) array.
    y: (k,) array.
    train: training trial indices.
    test: test trial indices. 
    '''
    x_train = x.reshape(-1, x.shape[1] * x.shape[2])[train]
    x_test = x.reshape(-1, x.shape[1] * x.shape[2])[test]
    y_train = y[train]
    y_test = y[test]
    ridge = Ridge(alpha=penalty_strength)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    if verbose:
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten()).statistic
        print(f'R2: {r2:.3f}, MSE: {mse:.3f}, Corr: {corr:.3f}')
    return y_train, y_test, y_pred


def sliding_window(
    data, 
    n_trials, 
    n_t_bins=30, 
    window_size=7, 
    aggregate=False
):
    '''
    Inputs:
    ------
    data: (k,c,t) array.
    '''
    data = data.transpose([1,0,2]) # (c, k, t)
    half_window_size = window_size // 2
    n_windows = n_t_bins - half_window_size + 1 \
        if window_size % 2 == 1 else n_t_bins - half_window_size + 2
    
    windowed_data = []
    for k in range(n_trials):
        for j in range(half_window_size, n_windows):
            window = [j-half_window_size, j+half_window_size-1] \
                if window_size % 2 == 0 else [j-half_window_size, j+half_window_size]
            if aggregate:
                data_per_window = data[:,k,window[0]:window[1]].sum(1)
            else:
                data_per_window = data[:,k,window[0]:window[1]].flatten()
            windowed_data.append(data_per_window)
    windowed_data = np.vstack(windowed_data)
    
    return windowed_data, half_window_size, n_windows


def sliding_window_decoder(
    x, 
    y, 
    train, 
    test, 
    window_size=7,
    penalty_strength=1000,
    verbose=True
):
    '''
    Inputs:
    ------
    x: (k,c,t) array.
    y: (k,) array.
    train: training trial indices.
    test: test trial indices. 
    '''
    n_trials = len(y)
    windowed_x, half_window_size, n_windows = sliding_window(x, n_trials, window_size=window_size)
    windowed_y = y[:, half_window_size:n_windows].reshape(-1, 1)
    
    x_by_trial = windowed_x.reshape((n_trials, -1))
    y_by_trial = windowed_y.reshape((n_trials, -1))
    x_train, x_test = x_by_trial[train], x_by_trial[test]
    y_train, y_test = y_by_trial[train], y_by_trial[test]

    x_train = x_train.reshape((-1, windowed_x.shape[1]))
    x_test = x_test.reshape((-1, windowed_x.shape[1]))
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    ridge = Ridge(alpha=penalty_strength)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    corr = pearsonr(y_test.flatten(), y_pred.flatten()).statistic
    if verbose:
        print(f'R2: {r2:.3f}, MSE: {mse:.3f}, Corr: {corr:.3f}')
    
    return y_train, y_test, y_pred, r2, mse, corr


def multi_trial_discrete_decoder(
    x, 
    y,
    train, 
    test,
    d = 5,
    penalty_type = "l2",
    penalty_strength = 1000,
    verbose = True
):
    x = x.transpose(1,2,0) # (n_c, n_t, n_k)
    n_k = x.shape[-1]
    
    # use spiking activity from [k-d, k+d] trials to 
    #     predict behavior in trial k.
    multi_trial_x = []
    for k in range(n_k):
        window = [k - d, k + d - 1] \
            if (2*d+1) % 2 == 0 else [k-d, k+d]
        if np.logical_and(window[0] >= 0, window[1] <= n_k):
            x_window = x[:,:,window[0]:window[1]].flatten()
        elif window[0] < 0:
            x_window = x[:,:,k:k+(2*d)].flatten()
        elif window[1] > n_k:
            x_window = x[:,:,k-(2*d):k].flatten()
        multi_trial_x.append(x_window)
    multi_trial_x = np.vstack(multi_trial_x)
    
    x_train = multi_trial_x[train]
    x_test = multi_trial_x[test]
    y_train = y[train]
    y_test = y[test]
    lr = LogisticRegression(random_state=seed, 
                            max_iter=1e4, 
                            tol = 0.01, 
                            solver='liblinear',
                            penalty=penalty_type, 
                            C=penalty_strength)
    lr.fit(x_train, y_train)
    y_prob = lr.predict_proba(x_test)
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_test, y_pred)
    if verbose:
        print(f'accuracy: {acc:.3f}')
    return y_train, y_test, y_pred, y_prob, acc
    
    
