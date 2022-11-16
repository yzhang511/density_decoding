import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error 
from sklearn.model_selection import KFold, StratifiedKFold

def decode_static(x_train, x_test, y_train, y_test, behave_type, seed=666):
    '''
    decode choice and stimulus
    to do: add prior
    '''
    
    if np.logical_or(behave_type == 'choice', behave_type == 'stimulus'):
        x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[-1])
        x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[-1])
        # decoder = SVC(random_state=seed, max_iter=1e4, tol = 0.01, 
        #               kernel='rbf', probability=True).fit(x_train, y_train.argmax(1))
        decoder = LogisticRegression(random_state=seed, max_iter=1e4, tol = 0.01).fit(x_train, y_train.argmax(1))
        probs = decoder.predict_proba(x_test)
        preds = probs.argmax(1)
        acc = accuracy_score(y_test.argmax(1), preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = np.nan
            print('only one class present in y_true, auc score is not defined in that case.')
      
    return acc, auc, preds, probs


def cv_decode_static(x, y, behave_type, n_folds=5, seed=666, shuffle=True):
    '''
    
    '''
    
    kf = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    
    fold = 0
    cv_accs = []; cv_aucs = []; cv_ids = []
    cv_obs = []; cv_preds = []; cv_probs = []
    for train, test in kf.split(x, y.argmax(1)):
        fold += 1
        acc, auc, preds, probs = decode_static(x[train], x[test], y[train], y[test], behave_type, seed)
        cv_ids.append(test)
        cv_obs.append(y[test])
        cv_accs.append(acc)
        cv_aucs.append(auc)
        cv_preds.append(preds)
        cv_probs.append(probs)
        print(f'{behave_type} fold {fold} test accuracy: {acc:.3f} auc: {auc:.3f}')
    filtered_cv_aucs = list(np.array(cv_aucs)[~np.isnan(cv_aucs)])
    print(f'{behave_type} mean of {fold}-fold cv accuracy: {np.mean(cv_accs):.3f} auc: {np.mean(filtered_cv_aucs):.3f}')
    print(f'{behave_type} sd of {fold}-fold cv accuracy: {np.std(cv_accs):.3f} auc: {np.std(filtered_cv_aucs):.3f}')
    
    return cv_accs, filtered_cv_aucs, cv_ids, cv_obs, cv_preds, cv_probs


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


def cv_decode_dynamic(x, y, n_trials, behave_type, n_splits=5, seed=666, shuffle=True):
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

    x_by_trial = x.reshape((n_trials, -1))
    y_by_trial = y.reshape((n_trials, -1))
   
    fold = 0
    cv_r2s = []; cv_rmses = []
    cv_ids = []; cv_obs = []; cv_preds = []
    for train, test in kf.split(x_by_trial):
        fold += 1
        x_train, x_test = x_by_trial[train], x_by_trial[test]
        y_train, y_test = y_by_trial[train], y_by_trial[test]

        x_train = x_train.reshape((-1, x.shape[1]))
        x_test = x_test.reshape((-1, x.shape[1]))
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        decoder = Ridge(alpha=2000)
        decoder.fit(x_train, y_train)
        y_pred = decoder.predict(x_test)
        cv_r2s.append(r2_score(y_test, y_pred))
        cv_rmses.append(mean_squared_error(y_test, y_pred))
        cv_ids.append(test)
        cv_obs.append(y_test.reshape(-1, y_by_trial.shape[1]))
        cv_preds.append(y_pred.reshape(-1, y_by_trial.shape[1]))
        print(f'{behave_type} fold {fold} test r2: {cv_r2s[-1]:.3f} rmse: {cv_rmses[-1]:.3f}')
        
    print(f'{behave_type} mean of {fold}-fold cv r2: {np.mean(cv_r2s):.3f} rmse: {np.mean(cv_rmses):.3f}')
    print(f'{behave_type} sd of {fold}-fold cv r2: {np.std(cv_r2s):.3f} rmse: {np.std(cv_rmses):.3f}')
    return cv_r2s, cv_rmses, cv_ids, cv_obs, cv_preds


