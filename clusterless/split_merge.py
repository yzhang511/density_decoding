import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.interpolate import UnivariateSpline
from .data_preprocess import load_kilosort_template_feature_mads

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


def calc_corr(u, v):
    return np.dot(u, v) / (np.linalg.norm(u)*np.linalg.norm(v))


def calc_corr_matrix(probs, tol=0.1):
    '''
    
    '''
    
    corr_mat = np.zeros((probs.shape[1], probs.shape[1]))
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            u = probs[:,i].copy()
            v = probs[:,j].copy()
            corr = calc_spike_corr(u, v)
            if i != j:
                corr_mat[i, j] = corr
            elif i == j:
                corr_mat[i, j] = 0  # exclude self-correlation
    return corr_mat


def calc_smooth_envelope_feature_mads(temp_amps, mad_xs, mad_zs, use_ks_template=False):
    '''
    monotonic smoothing envelopes for x and z. 
    to do: change ad hoc smoothing params.
    to do: remove using our features.
    '''
    
    # interpolate x
    offset = mad_xs.copy()
    if use_ks_template:   
        offset[:15] = offset[:15]+4.
        offset[15:20] = offset[15:20]+6.
        offset[20:] = offset[20:]+3.
    else:
        offset[:15] = offset[:15]+10.
        offset[15:20] = offset[15:20]+6.
        offset[20:] = offset[20:]+2.
    xs = np.linspace(temp_amps.min(), temp_amps.max(), 6)
    spl1 = UnivariateSpline(temp_amps, offset)
    spl2 = UnivariateSpline(xs, spl1(xs))
    xs2 = np.linspace(xs.min(), xs.max(), 40)
    envelope_xs = spl2(xs2)
    
    # interpolate z
    offset = mad_zs.copy()
    if use_ks_template: 
        offset[:15] = offset[:15]+8.
        offset[15:] = offset[15:]+4.
        zs = np.linspace(temp_amps.min(), temp_amps.max(), 4)
    else:
        offset[:15] = offset[:15]+15.
        offset[15:20] = offset[15:20]+10.
        offset[20:] = offset[20:]+3.
        zs = np.linspace(temp_amps.min(), temp_amps.max(), 6)
    spl1 = UnivariateSpline(temp_amps, offset)
    spl2 = UnivariateSpline(zs, spl1(zs))
    zs2 = np.linspace(zs.min(), zs.max(), 40)
    envelope_zs = spl2(zs2)

    return xs2, zs2, envelope_xs, envelope_zs


def split_criteria(data, labels, use_ks_template=False):
    '''
    to do: fix scale mismatch btw kilosort template features and our features.
    to do: remove using our features.
    '''
    
    avg_ptps = []
    mad_xs = []
    mad_zs = []
    for comp_idx in np.unique(labels):
        avg_ptp, mad_x, mad_z = calc_component_ptp_dependent_feature_mads(data, labels, comp_idx)
        avg_ptps.append(avg_ptp)
        mad_xs.append(mad_x)
        mad_zs.append(mad_z)
    avg_ptps = np.array(avg_ptps)
    mad_xs = np.array(mad_xs)
    mad_zs = np.array(mad_zs)
    
    if use_ks_template:
        binned_ptps, binned_xs, binned_zs = load_kilosort_template_feature_mads('data')
        xs, zs, envelope_xs, envelope_zs = \
        calc_smooth_envelope_feature_mads(binned_ptps, binned_xs, binned_zs, use_ks_template=use_ks_template)
        closest_bin_ids = [np.argmin(np.abs(binned_ptps - ptp)) for ptp in avg_ptps]    
    else:
        for bin_size in range(200):
            ptp_bins = np.linspace(np.min(avg_ptps), np.max(avg_ptps), bin_size) 
            ptp_masks = np.digitize(avg_ptps, ptp_bins, right=True)
            if len(np.unique(ptp_masks)) == 40:
                break
        binned_ptps = np.array([avg_ptps[ptp_masks == bin].mean() for bin in np.unique(ptp_masks)])
        binned_xs = np.array([mad_xs[ptp_masks == bin].mean() for bin in np.unique(ptp_masks)])
        binned_zs = np.array([mad_zs[ptp_masks == bin].mean() for bin in np.unique(ptp_masks)])
        closest_ptp_bins = [np.argmin(np.abs(ptp_bins - ptp)) for ptp in avg_ptps]
        closest_bin_ids = [np.argmin(np.abs(np.unique(ptp_masks) - bin_id)) for bin_id in closest_ptp_bins]
        xs, zs, envelope_xs, envelope_zs = \
        calc_smooth_envelope_feature_mads(binned_ptps, binned_xs, binned_zs, use_ks_template=use_ks_template)
        
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    axes[0].plot(binned_ptps, binned_xs, label='x', linestyle='dashed')
    axes[0].plot(xs, envelope_xs, label='envelope x', linewidth=2)
    axes[0].set_xlabel('template amplitude')
    axes[0].set_ylabel('feature MAD')
    axes[0].legend();
    axes[1].plot(binned_ptps, binned_zs, label='z', linestyle='dashed')
    axes[1].plot(zs, envelope_zs, label='envelope z', linewidth=2)
    axes[1].legend();
    axes[1].set_xlabel('template amplitude')
    plt.tight_layout()
    plt.show()
          
    split_ids = []
    for i in range(len(avg_ptps)):
        if np.logical_or(mad_xs[i] > envelope_xs[closest_bin_ids[i]], 
                         mad_xs[i] > envelope_zs[closest_bin_ids[i]]):
            split_ids.append(i)
            
    return split_ids


def split_gaussians(rootpath, sub_id, data, initial_gmm, initial_labels, split_ids, fit_model=False):
    '''
    
    '''
    gmm_name = f'{rootpath}/pretrained/{sub_id}/post_split_gmm'
    
    if fit_model:
        # before split
        init_bic = initial_gmm.bic(data)
        print(f'initial n_gaussians: {len(initial_gmm.means_)} bic: {round(init_bic, 2)}')

        pre_split_labels = set(np.unique(initial_labels)).difference(set(split_ids))
        print(f'keep {len(pre_split_labels)} gaussians and split {len(split_ids)} gaussians ...')

        pre_split_weights = np.vstack([initial_gmm.weights_[i] for i in pre_split_labels]).squeeze()
        pre_split_means = np.vstack([initial_gmm.means_[i] for i in pre_split_labels])
        pre_split_covariances = np.stack([initial_gmm.covariances_[i] for i in pre_split_labels])

        pre_split_gmm = GaussianMixture(n_components=len(pre_split_weights), covariance_type='full')
        pre_split_gmm.weights_ = pre_split_weights
        pre_split_gmm.means_ = pre_split_means
        pre_split_gmm.covariances_ = pre_split_covariances
        pre_split_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(pre_split_covariances))
        pre_split_bic = pre_split_gmm.bic(data)
        print(f'pre-split bic: {round(pre_split_bic, 2)}')

        # split
        post_split_weights = [pre_split_gmm.weights_]
        post_split_means = [pre_split_gmm.means_]
        post_split_covariances = [pre_split_gmm.covariances_]
        post_split_bics = [pre_split_bic, pre_split_bic]

        for i in split_ids:
            n_gaussians = 1
            while post_split_bics[-1] <= post_split_bics[-2]:
                n_gaussians += 1
                tmp_gmm = GaussianMixture(n_components=n_gaussians)
                tmp_gmm.fit(data[initial_labels == i])

                weights = np.hstack([pre_split_weights, tmp_gmm.weights_])
                means = np.vstack([pre_split_means, tmp_gmm.means_])
                covariances = np.vstack([pre_split_covariances, tmp_gmm.covariances_])

                new_gmm = GaussianMixture(n_components=len(weights), covariance_type='full')
                new_gmm.weights_ = weights
                new_gmm.means_ = means
                new_gmm.covariances_ = covariances
                new_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(new_gmm.covariances_))

                post_split_weights.append(new_gmm.weights_)
                post_split_means.append(new_gmm.means_)
                post_split_covariances.append(new_gmm.covariances_)

                new_bic = new_gmm.bic(data)
                post_split_bics.append(new_bic)
                print(f'split gaussian {i} into {n_gaussians} gaussians with updated bic: {round(new_bic, 2)}')

            pre_split_weights = post_split_weights[-2]
            pre_split_means = post_split_means[-2]
            pre_split_covariances = post_split_covariances[-2]
            post_split_bics = [new_bic, new_bic]

        # after split
        post_split_gmm = GaussianMixture(n_components=len(post_split_weights[-2]), covariance_type='full')
        post_split_gmm.weights_ = post_split_weights[-2]
        post_split_gmm.means_ = post_split_means[-2]
        post_split_gmm.covariances_ = post_split_covariances[-2]
        post_split_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(post_split_gmm.covariances_))
        
        np.save(gmm_name + '_weights', post_split_gmm.weights_, allow_pickle=False)
        np.save(gmm_name + '_means', post_split_gmm.means_, allow_pickle=False)
        np.save(gmm_name + '_covariances', post_split_gmm.covariances_, allow_pickle=False)
        
    else:
        means = np.load(gmm_name + '_means.npy')
        covar = np.load(gmm_name + '_covariances.npy')
        post_split_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
        post_split_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
        post_split_gmm.weights_ = np.load(gmm_name + '_weights.npy')
        post_split_gmm.means_ = means
        post_split_gmm.covariances_ = covar
        
    return post_split_gmm

def merge_criteria(corr_mat, threshold):
    '''
    to do: fix probe geometry sparsity issue.
    '''
    merge_ids = np.argwhere(corr_mat > threshold)
    return merge_ids



def merge_gaussians(rootpath, sub_id, data, post_split_gmm, post_split_labels, merge_ids, fit_model=False):
    '''
    
    '''
    gmm_name = f'{rootpath}/pretrained/{sub_id}/post_merge_gmm'
    
    if fit_model:
        # before merge
        init_bic = post_split_gmm.bic(data)
        print(f'initial n_gaussians: {len(post_split_gmm.means_)} bic: {round(init_bic, 2)}')

        pre_merge_labels = set(np.unique(post_split_labels)).difference(set(np.unique(merge_ids)))
        print(f'keep {len(pre_merge_labels)} gaussians and merge {len(merge_ids)} gaussians ...')

        pre_merge_weights = np.vstack([post_split_gmm.weights_[i] for i in pre_merge_labels]).squeeze()
        pre_merge_means = np.vstack([post_split_gmm.means_[i] for i in pre_merge_labels])
        pre_merge_covariances = np.stack([post_split_gmm.covariances_[i] for i in pre_merge_labels])

        pre_merge_gmm = GaussianMixture(n_components = len(pre_merge_weights), covariance_type='full')
        pre_merge_gmm.weights_ = pre_merge_weights
        pre_merge_gmm.means_ = pre_merge_means
        pre_merge_gmm.covariances_ = pre_merge_covariances
        pre_merge_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(pre_merge_covariances))
        pre_merge_bic = pre_merge_gmm.bic(X)
        print(f'pre-merge bic: {round(pre_merge_bic, 2)}')
        
        # merge
        post_merge_weights = [pre_merge_weights]
        post_merge_means = [pre_merge_means]
        post_merge_covariances = [pre_merge_covariances]

        for i, j in merge_ids:
            tmp_gmm = GaussianMixture(n_components=1)
            tmp_gmm.fit(np.vstack([data[post_split_labels == i], data[post_split_labels == j]]))

            weights = np.hstack([pre_merge_weights, tmp_gmm.weights_])
            means = np.vstack([pre_merge_means, tmp_gmm.means_])
            covariances = np.vstack([pre_merge_covariances, tmp_gmm.covariances_])

            new_gmm = GaussianMixture(n_components = len(weights), covariance_type='full')
            new_gmm.weights_ = weights
            new_gmm.means_ = means
            new_gmm.covariances_ = covariances
            new_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

            post_merge_weights.append(new_gmm.weights_)
            post_merge_means.append(new_gmm.means_)
            post_merge_covariances.append(new_gmm.covariances_)

            post_merge_bic = new_gmm.bic(data)
            print(f'merge pairs {i} and {j} with updated bic: {round(post_merge_bic, 2)}')

            pre_merge_weights = post_merge_weights[-1]
            pre_merge_means = post_merge_means[-1]
            pre_merge_covariances = post_merge_covariances[-1]
        
        # after merge
        post_merge_gmm = GaussianMixture(n_components=len(post_merge_weights[-1]), covariance_type='full')
        post_merge_gmm.weights_ = post_merge_weights[-1]
        post_merge_gmm.means_ = post_merge_means[-1]
        post_merge_gmm.covariances_ = post_merge_covariances[-1]
        post_merge_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(post_merge_gmm.covariances_))
        
        np.save(gmm_name + '_weights', post_merge_gmm.weights_, allow_pickle=False)
        np.save(gmm_name + '_means', post_merge_gmm.means_, allow_pickle=False)
        np.save(gmm_name + '_covariances', post_merge_gmm.covariances_, allow_pickle=False)
    else:
        means = np.load(gmm_name + '_means.npy')
        covar = np.load(gmm_name + '_covariances.npy')
        post_merge_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
        post_merge_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
        post_merge_gmm.weights_ = np.load(gmm_name + '_weights.npy')
        post_merge_gmm.means_ = means
        post_merge_gmm.covariances_ = covar
        
    return post_merge_gmm
        