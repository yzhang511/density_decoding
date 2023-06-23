import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score
from density_decoding.utils.utils import get_odd_number


def generic_decoder(
    x, 
    y, 
    train, 
    test,
    behavior_type,
    penalty_strength=1,
    verbose=False,
    seed=666
):
    """
    Decode dicrete and continuous behaviors w/ L2-penalized linear models.
    
    Args:
        x: size (n_k, n_c, n_t) array (spike count or weight matrix) 
        y: size (n_k,) array.
        train: training trial index.
        test: test trial index. 
        behavior_type: "discrete" or "continuous"
        penalty_strength: larger values imply stronger regularization
        verbose: whether to print decoding results
        
    Returns:
        y_train: (n_k,) or (n_k, n_t) array
        y_test: (n_k,) or (n_k, n_t) array
        y_pred: (n_k,) or (n_k, n_t) array
        metrics: a dict where decoding results are saved
    """
    if len(x.shape) == 3:
        x = x.reshape(-1, x.shape[1] * x.shape[-1])
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]
    
    metrics = {}
    if behavior_type == "discrete":
        lr = LogisticRegression(random_state=seed, 
                                max_iter=1e4, 
                                tol = 0.01, 
                                solver='liblinear',
                                penalty="l2", 
                                C=1/penalty_strength)
        lr.fit(x_train, y_train)

        y_prob = lr.predict_proba(x_test)
        y_pred = y_prob.argmax(1)
        metrics.update({"acc": accuracy_score(y_test, y_pred)})

        if verbose:
            print(f'accuracy: {metrics["acc"]:.3f}')
      
    elif behavior_type == "continuous":
        ridge = Ridge(alpha=penalty_strength)
        ridge.fit(x_train, y_train)
        
        y_pred = ridge.predict(x_test)
        metrics.update({"r2": r2_score(y_test, y_pred)})
        metrics.update({"mse": mean_squared_error(y_test, y_pred)})
        metrics.update({"corr": pearsonr(y_test.flatten(), 
                                         y_pred.flatten()).statistic})

        if verbose:
            print(f'R2: {metrics["r2"]:.3f}, MSE: {metrics["mse"]:.3f}, Corr: {metrics["corr"]:.3f}')

    return y_train, y_test, y_pred, metrics

    
def sliding_window(x, window_size):
    """
    Convert spike data into a suitable format for decoding.
    
    Args:
        x: size (n_k, n_c, n_t) array (spike count or weight matrix)
        window_size: int (less than n_t / 2)
        
    Returns:
        windowed_x: converted spike data
        half_window_size: int (= window_size / 2)
        n_windows: int (number of windows)
    """
    
    n_k, n_c, n_t = x.shape
    # (n_k, n_c, n_t) --> (n_c, n_k, n_t)
    x = x.transpose([1,0,2]) 
    half_window_size = window_size // 2
    n_windows = n_t - half_window_size + 1 \
                if window_size % 2 == 1 else n_t - half_window_size + 2
    
    windowed_x = []
    for k in range(n_k):
        for j in range(half_window_size, n_windows):
            window = [j-half_window_size, j+half_window_size-1] \
                     if window_size % 2 == 0 else [j-half_window_size, j+half_window_size]
            sub_x = x[:, k, window[0]: window[1]].flatten()
            windowed_x.append(sub_x)
    windowed_x = np.vstack(windowed_x)
    
    return windowed_x, half_window_size, n_windows


def sliding_window_decoder(
    x, 
    y, 
    train, 
    test, 
    behavior_type,
    window_size=7,
    penalty_strength=1000,
    verbose=True,
    seed=666
):
    """
    Decode continuous and discrete behaviors via sliding window algorithm.
        e.g., use data from [t-d, t+d] to decode behavior at time t, d = half window size
        e.g., use data from [k-d, k+d] to decode behavior in trial k, d = half window size
    
    Args:
        x: size (n_k, n_c, n_t) array (spike count or weight matrix) 
        y: size (n_k,) array.
        train: training trial index.
        test: test trial index. 
        behavior_type: "discrete" or "continuous"
        window_size: size of the sliding window
        penalty_strength: larger values imply stronger regularization
        verbose: whether to print decoding results
        
    Returns:
        y_train: (n_k,) or (n_k, n_t) array
        y_test: (n_k,) or (n_k, n_t) array
        y_pred: (n_k,) or (n_k, n_t) array
        metrics: a dict where decoding results are saved
    """
    
    n_k, _, _ = x.shape
    if behavior_type == "continuous":
        
        # convert spike data and behavior via sliding window
        windowed_x, half_window_size, n_windows = sliding_window(
            x, window_size=window_size
        )
        windowed_y = y[:, half_window_size:n_windows].reshape(-1, 1)

        # reshape converted data for decoding 
        x_by_trial = windowed_x.reshape(n_k, -1)
        y_by_trial = windowed_y.reshape(n_k, -1)
        x_train, x_test = x_by_trial[train], x_by_trial[test]
        y_train, y_test = y_by_trial[train], y_by_trial[test]

        x_train = x_train.reshape(-1, windowed_x.shape[1])
        x_test = x_test.reshape(-1, windowed_x.shape[1])
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        ridge = Ridge(alpha=penalty_strength)
        ridge.fit(x_train, y_train)
        y_pred = ridge.predict(x_test)
        
        metrics = {}
        metrics.update({"r2": r2_score(y_test, y_pred)})
        metrics.update({"mse": mean_squared_error(y_test, y_pred)})
        metrics.update({"corr": pearsonr(y_test.flatten(), 
                                         y_pred.flatten()).statistic})

        if verbose:
            print(f'R2: {metrics["r2"]:.3f}, MSE: {metrics["mse"]:.3f}, Corr: {metrics["corr"]:.3f}')
        
    elif behavior_type == "discrete":
        
        half_window_size = get_odd_number(window_size)
        # (n_k, n_c, n_t) --> (n_c, n_t, n_k)
        x = x.transpose(1,2,0) 

        windowed_x = []
        for k in range(n_k):
            window = [k-half_window_size, k+half_window_size+1] 
            if np.logical_and(window[0] >= 0, window[1] <= n_k):
                sub_x = x[:,:,window[0]:window[1]].flatten()
            elif window[0] < 0:
                sub_x = x[:,:,k:k+2*half_window_size+1].flatten()
            elif window[1] > n_k:
                sub_x = x[:,:,k-2*half_window_size:k+1].flatten()
            windowed_x.append(sub_x)
        windowed_x = np.vstack(windowed_x)
        
        y_train, y_test, y_pred, metrics = generic_decoder(
            windowed_x, 
            y, 
            train, 
            test,
            behavior_type,
            penalty_strength=penalty_strength,
            verbose=verbose,
            seed=seed
        )
            
    return y_train, y_test, y_pred, metrics

