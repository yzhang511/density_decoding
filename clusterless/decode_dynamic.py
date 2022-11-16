import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def decode_dynamic(x_train, x_test, y_train, y_test, behave_type, seed=666):
    '''

    '''
    
    if behave_type in ['motion energy', 'wheel velocity', 'wheel speed',
                       'paw speed', 'nose speed', 'pupil diameter']:
        decoder = Ridge(alpha=2e3).fit(x_train, y_train)
        preds = decoder.predict(x_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
    return r2, rmse, preds


def cv_decode_dynamic(x, y, behave_type, n_folds=5, seed=666, shuffle=True):
    '''
    
    '''
    
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    
    fold = 0
    cv_r2s = []; cv_rmses = []
    cv_ids = []; cv_obs = []; cv_preds = []
    for train, test in kf.split(x, y):
        fold += 1
        r2, rmse, preds = decode_dynamic(x[train], x[test], y[train], y[test], behave_type, seed)
        cv_r2s.append(r2)
        cv_rmses.append(rmse)
        cv_ids.append(test)
        cv_obs.append(y[test])
        cv_preds.append(preds)
        print(f'{behave_type} fold {fold} test r2: {r2:.3f} r2: {r2:.3f}: {rmse:.3f}')
    print(f'{behave_type} mean of {fold}-fold cv r2: {np.mean(cv_r2s):.3f} rmse: {np.mean(cv_rmses):.3f}')
    print(f'{behave_type} sd of {fold}-fold cv r2: {np.std(cv_r2s):.3f} rmse: {np.std(cv_rmses):.3f}')
    
    return cv_r2s, cv_rmses, cv_ids, cv_obs, cv_preds
