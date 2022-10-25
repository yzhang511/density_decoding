import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

def tpca(neural_data, n_projs=15):
    '''
    
    '''
    
    n_trials, n_units, n_time_bins = neural_data.shape
    neural_data = neural_data.reshape(n_trials*n_units, n_time_bins)
    
    # standardize before pca
    scaler = StandardScaler()
    scaler.fit(neural_data)
    std_neural_data = scaler.transform(neural_data)
    mu = np.mean(std_neural_data, axis=0)
    
    # pca projections
    pca = PCA(n_components=n_projs, svd_solver='full')
    proj_neural_data = pca.fit_transform(std_neural_data)
    
    # pca reconstructions
    recon_neural_data = np.dot(proj_neural_data, pca.components_)
    recon_neural_data += mu
    recon_neural_data = recon_neural_data.reshape(n_trials, n_units, n_time_bins)
    
    proj_neural_data = proj_neural_data.reshape(n_trials, n_units, n_projs)
    
    return proj_neural_data, recon_neural_data