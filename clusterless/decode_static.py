import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score 
from sklearn.model_selection import KFold, StratifiedKFold

def decode_static(x_train, x_test, y_train, y_test, behave_type, seed=666):
    '''
    decode choice and stimulus
    to do: add prior
    '''
    
    if np.logical_or(behave_type == 'choice', behave_type == 'stimulus'):
        x_train = x_train.reshape(-1, x_train[1]*x_train[-1])
        x_test = x_test.reshape(-1, x_test[1]*x_test[-1])
        decoder = SVC(random_state=seed, max_iter=1e4, tol = 0.01, kernel='rbf', probability=True).fit(x_train, y_train.argmax(1))
        probs = clf.predict_proba(x_test)
        preds = probs.argmax(1)
        acc = accuracy_score(y_test.argmax(1), preds)
        auc = roc_auc_score(y_test, probs)
      
    return acc, auc, preds, probs


def cv_decode_static(x, y, behave_type, n_folds=5, seed=666, shuffle=True):
    '''
    
    '''
    
    kf = KFold(n_splits=n_folds, random_state=seed, shuffle=shuffle)
    
    fold = 1
    cv_accs = []; cv_aucs = []; cv_ids = []
    cv_obs = []; cv_preds = []; cv_probs = []
    for train, test in kf.split(x):
        fold += 1
        acc, auc, preds, probs = decode_static(x[train], x[test], y[train], y[test], behave_type, seed)
        cv_ids.append(test)
        cv_obs.append(y[test])
        cv_accs.append(acc)
        cv_aucs.append(auc)
        cv_preds.append(preds)
        cv_probs.append(probs)
        print(f'{behave_type} fold {fold} test accuracy: {acc:.3f} auc: {auc:.3f}')
    print(f'{behave_type} mean of {fold}-fold cv accuracy: {np.mean(cv_accs):.3f} auc: {np.mean(cv_aucs):.3f}')
    print(f'{behave_type} sd of {fold}-fold cv accuracy: {np.std(cv_accs):.3f} auc: {np.std(cv_aucs):.3f}')
        
    return cv_accs, cv_aucs, cv_ids, cv_obs, cv_preds, cv_probs