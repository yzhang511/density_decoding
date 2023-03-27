import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


def dynamic_decoder(x, y, train, test):
    '''
    
    '''
    x_train = x.reshape(-1, x.shape[1] * x.shape[2])[train]
    x_test = x.reshape(-1, x.shape[1] * x.shape[2])[test]
    y_train = y[train]
    y_test = y[test]

    ridge = Ridge(alpha=1000)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    
    # print(f'R2: {r2_score(y_test, y_pred):.3f}, MSE: {mean_squared_error(y_test, y_pred):.3f}, Corr: {pearsonr(y_test.flatten(), y_pred.flatten()).statistic:.3f}')
    
    return y_train, y_test, y_pred


def sliding_window(
    neural_data, 
    n_trials, 
    n_time_bins = 30, 
    window_size = 7, 
    aggregate = False
):
    '''
    to do: assert input shape of neural data 
    '''
    neural_data = neural_data.transpose([1,0,2]) # (n_units, n_trials, n_time_bins)
    
    half_window_size = window_size // 2
    n_windows = n_time_bins - half_window_size + 1 \
        if window_size % 2 == 1 else n_time_bins - half_window_size + 2
    
    windowed_neural_data = []
    for i in range(n_trials):
        for j in range(half_window_size, n_windows):
            window = [j-half_window_size, j+half_window_size-1] \
                if window_size % 2 == 0 else [j-half_window_size, j+half_window_size]
            if aggregate:
                neural_data_per_window = neural_data[:,i,window[0]:window[1]].sum(1)
            else:
                neural_data_per_window = neural_data[:,i,window[0]:window[1]].flatten()
            windowed_neural_data.append(neural_data_per_window)

    windowed_neural_data = np.vstack(windowed_neural_data)

    return windowed_neural_data, half_window_size, n_windows


def sliding_window_decoder(x, y, train, test):
    '''
    
    '''
    n_trials = len(y)
    windowed_x, half_window_size, n_windows = sliding_window(x, n_trials, window_size = 7)
    windowed_y = y[:,half_window_size:n_windows].reshape(-1,1)
    
    x_by_trial = windowed_x.reshape((n_trials, -1))
    y_by_trial = windowed_y.reshape((n_trials, -1))
    x_train, x_test = x_by_trial[train], x_by_trial[test]
    y_train, y_test = y_by_trial[train], y_by_trial[test]

    x_train = x_train.reshape((-1, windowed_x.shape[1]))
    x_test = x_test.reshape((-1, windowed_x.shape[1]))
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    ridge = Ridge(alpha=1000)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    corr = pearsonr(y_test.flatten(), y_pred.flatten()).statistic
    
    print(f'R2: {r2:.3f}, MSE: {mse:.3f}, Corr: {corr:.3f}')
    
    return y_train, y_test, y_pred, r2, mse, corr

