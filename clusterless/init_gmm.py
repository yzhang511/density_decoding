import numpy as np
import random
from sklearn.mixture import GaussianMixture

def initial_gaussian_mixtures(rootpath, sub_id, trials, n_gaussians=300, seed=666, fit_model=True):
    '''
    
    '''
    
    gmm_name = f'{rootpath}/pretrained/{sub_id}/initial_gmm'
    
    if fit_model:
        trials_ids = np.arange(len(trials))
        random.seed(seed)
        random.shuffle(trials_ids)
        shuffled_unsorted = np.vstack([trials[i] for i in trials_ids])[:,1:]
    
        gmm = GaussianMixture(n_components=n_gaussians)
        gmm.fit(shuffled_unsorted)
        np.save(gmm_name + '_weights', gmm.weights_, allow_pickle=False)
        np.save(gmm_name + '_means', gmm.means_, allow_pickle=False)
        np.save(gmm_name + '_covariances', gmm.covariances_, allow_pickle=False)
        
    means = np.load(gmm_name + '_means.npy')
    covar = np.load(gmm_name + '_covariances.npy')
    loaded_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(gmm_name + '_weights.npy')
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    
    return loaded_gmm