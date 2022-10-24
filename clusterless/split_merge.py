import numpy as np
import random
from sklearn.mixture import GaussianMixture


def calc_mad(arr):
    """ 
    median absolute deviation: a "robust" version of standard deviation.
    """
    arr = np.ma.array(arr).compressed() 
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def calc_component_ptp_dependent_feature_mads(data, labels, comp_idx):
    avg_ptp = data[labels == comp_idx][:,-1].mean()
    mad_x = calc_mad(data[labels == comp_idx][:,0])
    mad_z = calc_mad(data[labels == comp_idx][:,1])
    return avg_ptp, mad_x, mad_z

def calc_smooth_envelope_feature_mads():
    return


def split_criteria(data, labels, template_amps, envelope_xs, envelope_zs):
    '''
    
    '''
    
    avg_ptps = []
    mad_xs = []
    mad_zs = []
    for comp_idx in np.unique(labels):
        avg_ptp, mad_x, mad_z = calc_component_ptp_dependent_feature_mads(data, labels, comp_idx)
        avg_ptps.append(avg_ptp)
        mad_xs.append(mad_x)
        mad_zs.append(mad_z)
    mad_xs = np.array(mad_xs)
    mad_zs = np.array(mad_zs)
    
    ptp_bins = [np.argmin(np.abs(template_amps - ptp)) for ptp in avg_ptps] # check this
    std_mad_xs = (mad_xs - mad_xs.min()) / (mad_xs.max() - mad_xs.min())
    std_mad_zs = (mad_zs - mad_zs.min()) / (mad_zs.max() - mad_zs.min())
    std_envelope_xs = (envelope_xs - envelope_xs.min()) / (envelope_xs.max() - envelope_xs.min())
    std_envelope_zs = (envelope_zs - envelope_zs.min()) / (envelope_zs.max() - envelope_zs.min())

    split_ids = []
    for i in range(len(avg_ptps)):
        if np.logical_or(std_mad_xs[i] > std_envelope_xs[ptp_bins[i]], 
                         std_mad_zs[i] > std_envelope_zs[ptp_bins[i]]):
            split_ids.append(i)
    return split_ids

def split_gaussians(data, initial_gmm, initial_labels, split_ids):
    '''
    
    '''
    init_bic = initial_gmm.bic(data)
    print(f'initial n_gaussians: {len(initial_gmm.means_)} bic: {round(init_bic, 2)}')

    pre_split_labels = set(np.unique(initial_labels)).difference(set(split_ids))
    print(f'keep {len(pre_split_labels)} gaussians and split {len(split_ids)} gaussians ...')

    weights = np.vstack([initial_gmm.weights_[i] for i in pre_split_labels]).squeeze()
    means = np.vstack([initial_gmm.means_[i] for i in pre_split_labels])
    covariances = np.stack([initial_gmm.covariances_[i] for i in pre_split_labels])

    pre_split_gmm = GaussianMixture(n_components=len(gmm_weights), covariance_type='full')
    pre_split_gmm.weights_ = weights
    pre_split_gmm.means_ = means
    pre_split_gmm.covariances_ = covariances
    pre_split_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))
    pre_split_bic = pre_split_gmm.bic(data)
    print(f'pre-split bic: {round(pre_split_bic, 2)}')
    
    
    
    
    return pre_split_gmm, init_bic, pre_split_bic
