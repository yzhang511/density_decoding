import numpy as np
import random
from sklearn.mixture import GaussianMixture
import isosplit

def initialize_gmm(data):
    '''
    
    '''
    
    sub_weights_lst = []
    sub_means_lst = []
    sub_covs_lst = []

    for channel in np.unique(data[:,0]):
        sub_s = data[data[:,0] == channel, 1:]

        if sub_s.shape[0] > 10:
            try:
                isosplit_labels = isosplit.isosplit(sub_s.T, K_init=20, min_cluster_size=10,
                                                    whiten_cluster_pairs=1, refine_clusters=1)
            except AssertionError:
                continue
            except ValueError:
                continue
        elif sub_s.shape[0] < 2:
            continue
        else:
            sub_gmm = GaussianMixture(n_components=1, 
                                  covariance_type='full',
                                  init_params='k-means++', 
                                  verbose=0)
            sub_gmm.fit(sub_s)
            sub_labels = sub_gmm.predict(sub_s)
            sub_weights = len(sub_labels)/len(data)
            sub_weights_lst.append(sub_weights)
            sub_means_lst.append(sub_gmm.means_)
            sub_covs_lst.append(sub_gmm.covariances_)
            continue

        n_splits = np.unique(isosplit_labels).shape[0]
        # print(f'channel {channel} has {n_splits} modes ...')

        if n_splits == 1: 
            sub_gmm = GaussianMixture(n_components=1, 
                                  covariance_type='full',
                                  init_params='k-means++', 
                                  verbose=0)
            sub_gmm.fit(sub_s)
            sub_labels = sub_gmm.predict(sub_s)
            sub_weights = len(sub_labels)/len(data)
            sub_weights_lst.append(sub_weights)
            sub_means_lst.append(sub_gmm.means_)
            sub_covs_lst.append(sub_gmm.covariances_)
        else:
            for label in np.arange(n_splits):
                mask = isosplit_labels == label
                sub_gmm = GaussianMixture(n_components=1, 
                                  covariance_type='full',
                                  init_params='k-means++', 
                                  verbose=0)
                sub_gmm.fit(sub_s[mask])
                sub_labels = sub_gmm.predict(sub_s[mask])
                sub_weights = len(sub_labels)/len(data)
                sub_weights_lst.append(sub_weights)
                sub_means_lst.append(sub_gmm.means_)
                sub_covs_lst.append(sub_gmm.covariances_)

    sub_weights = np.hstack(sub_weights_lst)
    sub_means = np.vstack(sub_means_lst)
    sub_covs = np.vstack(sub_covs_lst)
    
    gmm = GaussianMixture(n_components=len(sub_weights), covariance_type='full', init_params='k-means++')
    gmm.weights_ = sub_weights
    gmm.means_ = sub_means
    gmm.covariances_ = sub_covs
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(sub_covs))
    
    
    return gmm


