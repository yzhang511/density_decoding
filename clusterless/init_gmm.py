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


def initialize_gaussian_mixtures_with_spike_sorting(rootpath, sub_id, trials, seed=666, fit_model=True):
    '''
    
    '''
    
    init_spikes_labels = np.load(f'{rootpath}/{sub_id}/spike_sorting_results/spikes_indices.npy')[:,1]
    init_spikes_features = np.load(f'{rootpath}/{sub_id}/spike_sorting_results/spikes_features.npy')[:,[0,2,4]]
    
    clustered_features = [init_spikes_features[np.argwhere(init_spikes_labels == k)].squeeze() 
                              for k in np.unique(init_spikes_labels)]
    clustered_weights = np.array([len(clustered_features[k])/len(init_spikes_labels) for k in np.unique(init_spikes_labels)])
    clustered_means = np.array([[clustered_features[k][:,0].mean(),
                                 clustered_features[k][:,1].mean(),
                                 clustered_features[k][:,2].mean()]
                                 for k in np.unique(init_spikes_labels)])
    # use diagonal matrix to seed gmm - empirical covs not symmetric p.d.
    clustered_covs = np.array([np.diag(np.diag(np.cov(clustered_features[k].transpose()))) 
                                   for k in np.unique(init_spikes_labels)])
    
    gmm_name = f'{rootpath}/pretrained/{sub_id}/sorting_initialized_gmm'
    
    if fit_model:
        trials_ids = np.arange(len(trials))
        random.seed(seed)
        random.shuffle(trials_ids)
        shuffled_unsorted = np.vstack([trials[i] for i in trials_ids])[:,1:]
    
        gmm = GaussianMixture(n_components=len(np.unique(init_spikes_labels)), 
                             weights_init=clustered_weights,
                             means_init=clustered_means,
                             precisions_init=np.linalg.cholesky(np.linalg.inv(clustered_covs)))
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