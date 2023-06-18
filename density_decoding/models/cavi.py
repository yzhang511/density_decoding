import numpy as np
from tqdm import tqdm
import torch
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, roc_auc_score
from density_decoding.utils.utils import safe_log, safe_divide

class CAVI():
    def __init__(
        self, 
        init_means, 
        init_covs, 
        init_lambdas, 
        train_trial_idxs, 
        train_time_idxs, 
        test_trial_idxs, 
        test_time_idxs
    ):
        """
        CAVI model that only handles binary decoding variables. 
        
        Args:
            init_means: size (n_c, n_d) array,
                        n_c = number of Gaussian mixture components, 
                        n_d = spike feature dim
            init_covs: size (n_c, n_d, n_d) array
            init_lambdas: size (n_c, n_t, n_p) array, 
                          n_t = number of time bins, 
                          n_p = number of categories of the behavior variable
            train_trial_idxs: a list of arrays that contains the trial index of each spike
            train_time_indices: a list of arrays that contains the time index of each spike
            test_trial_idxs: a list of arrays that contains the trial index of each spike
            test_time_idxs: a list of arrays that contains the time index of each spike
        """
        
        self.train_n_k = len(train_trial_idxs)
        self.test_n_k = len(test_trial_idxs)
        self.n_t = len(train_time_idxs)
        self.n_c, self.n_d = init_means.shape
        self.init_mu = torch.tensor(init_means)
        self.init_cov = torch.tensor(init_covs)
        self.init_lam = torch.tensor(init_lambdas)
        self.train_ks = train_trial_idxs
        self.train_ts = train_time_idxs
        self.test_ks = test_trial_idxs
        self.test_ts = test_time_idxs
        
        
    def _compute_gmm_log_pdf(self, s, mu, cov, safe_cov=None):
        """
        Compute the log-likelihood of the mixture model. 
        
        Args:
            s: size (N, n_d) array, N = number of spikes, n_d = spike feature dim
            mu: size (n_c, n_d) array (updated mixture means)
            cov: size (n_c, n_d, n_d) array (updated mixture covariance matrix)
            safe_cov: size (n_c, n_d, n_d) array
                      (alternative covariance matrix to use in case cov is non-PSD)
         
        Returns:
            ll: size (N, n_c) array; computed log-likelihood 
        """
        
        ll = []
        for c in range(self.n_c):
            try:
                ll.append(
                    torch.tensor(
                        multivariate_normal.logpdf(s, mu[c], cov[c])
                    )
                )
            except np.linalg.LinAlgError:
                # TO DO: Need a better solution. 
                # This error occurs when the covariance matrix is not PSD.
                # We can use the initial covariance matrix as a replacement to ensure numerical stability.
                ll.append(
                    torch.tensor(
                        multivariate_normal.logpdf(s, mu[c], safe_cov[c])
                    )
                )
                
        return torch.vstack(ll).T 
    
    
    def _compute_encoder_elbo(self, r, y, ll, norm_lam):
        """
        Compute the ELBO for the encoder model.
        
        Args:
            r: size (N, n_c) array (normalized E_q(z)[z]) 
            y: size (N, n_p) array (convenient rep of observed y for einsum) 
            ll: size (N, n_c) array (computed log-likelihood)
            norm_lam: size (n_c, n_t, n_p) array (normalized lambda)
        
        Returns:
            elbo: float; ELBO
        """
        
        elbo = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,c], ll[:,c]) for c in range(self.n_c)]
        ))
        
        elbo += torch.tensor(
            [ torch.einsum(
                'ij,il,j->', r[self.train_ts[t]], y[self.train_ts[t]], norm_lam[:,t,1]
              ) + \
              torch.einsum(
                'ij,il,j->', r[self.train_ts[t]], 1-y[self.train_ts[t]], norm_lam[:,t,0]
              ) for t in range(self.n_t) 
            ]).sum()
        
        elbo -= torch.einsum('ij,ij->', safe_log(r), r)
        
        return elbo
    
    
    def _compute_decoder_elbo(self, r, ll, norm_lam, nu, nu_k, p):
        """
        Compute the ELBO for the decoder model.
        
        Args:
            r: size (N, n_c) array (normalized E_q(z)[z]) 
            ll: size (N, n_c) array (computed log-likelihood)
            norm_lam: size (n_c, n_t, n_p) array (normalized lambda) 
            nu: size (N, n_p) array (convenient rep of E_q(y)[y] for einsum) 
            nu_k: size (n_k,) array (alternative rep of E_q(y)[y])
            p: float; prob. of choosing 0 or 1 for the binary variable
        
        Returns:
            elbo: float; ELBO
        """
        
        elbo = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,c], ll[:,c]) for c in range(self.n_c)]
        ))
        
        elbo += torch.tensor(
            [ torch.einsum(
                'ij,il,j->', r[self.test_ts[t]], nu[self.test_ts[t]], norm_lam[:,t,1]
              ) + \
              torch.einsum(
                'ij,il,j->', r[self.test_ts[t]], 1-nu[self.test_ts[t]], norm_lam[:,t,0]
              ) for t in range(self.n_t) 
            ]).sum()
        
        elbo += torch.sum(nu_k * safe_log(p) + (1-nu_k) * safe_log(1-p))
        elbo -= torch.einsum('ij,ij->', safe_log(r), r)
        elbo -= torch.sum(safe_log(nu_k) * nu_k)
        
        return elbo
    
    
    def _encode_e_step(self, r, y, ll, norm_lam):
        """
        Execute the E step of the encoder.
        
        Args:
            r: size (N, n_c) array (normalized E_q(z)[z])
            y: size (N, n_p) array (convenient rep of observed y for einsum)
            ll: size (N, n_c) array (computed log-likelihood) 
            norm_lam: size (n_c, n_t, n_p) array (normalized lambda)
        
        Returns:
            r: size (N, n_c) array (updated normalized E_q(z)[z])  
        """
        
        for t in range(self.n_t):
            r[self.train_ts[t]] = torch.exp( 
                  ll[self.train_ts[t]] + \
                  torch.einsum('il,j->ij', y[self.train_ts[t]], norm_lam[:,t,1]) + \
                  torch.einsum('il,j->ij', 1-y[self.train_ts[t]], norm_lam[:,t,0])
            )
            r[self.train_ts[t]] = torch.einsum(
                'ij,i->ij', r[self.train_ts[t]], 1/r[self.train_ts[t]].sum(1)
            )
            
        return r
        
    
    def _encode_m_step(self, s, r, y, mu, lam):
        """
        Execute the M step of the encoder.
        
        Args:
            s: size (N, n_d) array, N = number of spikes, n_d = spike feature dim
            r: size (N, n_c) array (normalized E_q(z)[z])  
            y: size (N, n_p) array (convenient rep of observed y for einsum)
            mu: size (n_c, n_d) array (GMM means)
            lam: size (n_c, n_t, n_p) array (unnormalized lambda)
        
        Returns:
            mu: size (n_c, n_d) array (updated GMM means) 
            cov: size (n_c, n_d, n_d) array (updated GMM covariance matrix)
            lam: size (n_c, n_t, n_p) array (updated unnormalized lambda)
            norm_lam: size (n_c, n_t, n_p) array (updated normalized lambda)
        """
        
        for c in range(self.n_c):
            no_c_idx = torch.cat([torch.arange(c), torch.arange(c+1, self.n_c)])
            lam_sum_no_c = lam[no_c_idx,:,:].sum(0)
            for t in range(self.n_t):
                num1 = torch.einsum('i,il,->', r[self.train_ts[t],c], y[self.train_ts[t]], lam_sum_no_c[t,1])
                denom1 = np.einsum('ij,il->', r[self.train_ts[t]][:,no_c_idx], y[self.train_ts[t]])
                num0 = torch.einsum('i,il,->', r[self.train_ts[t],c], 1-y[self.train_ts[t]], lam_sum_no_c[t,0])
                denom0 = np.einsum('ij,il->', r[self.train_ts[t]][:,no_c_idx], 1-y[self.train_ts[t]])
                lam[c,t,1], lam[c,t,0] = num1 / denom1, num0 / denom0
                
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
                    ',i,ip,id->pd', 1/norm[c], r[:,c], s-mu[c], s-mu[c]) for c in range(self.n_c)]
        
        return mu, cov, lam, norm_lam
    
    
    def _decode_e_step(self, r, ll, norm_lam, nu, nu_k, p):
        """
        Execute the E step of the decoder.
        
        Args:
            r: size (N, n_c) array (normalized E_q(z)[z])  
            ll: size (N, n_c) array (computed log-likelihood) 
            norm_lam: size (n_c, n_t, n_p) array (normalized lambda)
            nu: size (N, n_p) array (convenient rep of E_q(y)[y] for einsum)
            nu_k: size (n_k,) array (alternative rep of E_q(y)[y])
            p: float; prob. of choosing 0 or 1 for the binary variable. 
        
        Returns:
            r: size (N, n_c) array (updated normalized E_q(z)[z])  
            nu: size (N, n_p) array (updated E_q(y)[y] for einsum) 
            nu_k: size (n_k,) array (updated E_q(y)[y])
        """
        
        for t in range(self.n_t):
            r[self.test_ts[t]] = torch.exp( ll[self.test_ts[t]] + \
                      torch.einsum('il,j->ij', nu[self.test_ts[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-nu[self.test_ts[t]], norm_lam[:,t,0])
            )
            r[self.test_ts[t]] = torch.einsum(
                'ij,i->ij', r[self.test_ts[t]], 1/r[self.test_ts[t]].sum(1)
            )
            
        for k in range(self.test_n_k):
            y_tilde0, y_tilde1 = safe_log(1-p), safe_log(p)
            for t in range(self.n_t):
                k_t_idx = np.intersect1d(self.test_ks[k], self.test_ts[t])
                y_tilde0 += torch.einsum('ij,j->', r[k_t_idx], norm_lam[:,t,0])
                y_tilde1 += torch.einsum('ij,j->', r[k_t_idx], norm_lam[:,t,1])
                
            # TO DO: Need a better solution. 
            # exp(y_tilde) explodes to 0 so need to offset to ensure numerical stability.
            offset = 1. / (torch.min(torch.tensor([y_tilde0, y_tilde1])) / -745.) 
            y_tilde0, y_tilde1 = torch.exp(y_tilde0 * offset), torch.exp(y_tilde1 * offset)
            nu_k[k] = safe_divide(y_tilde1, y_tilde0+y_tilde1)
            nu[self.test_ks[k]] = nu_k[k]
            
        return r, nu, nu_k
    
    
    def _decode_m_step(self, s, r, nu_k, mu):
        """
        Execute the M step of the decoder.
        
        Args:
            s: size (N, n_d) array, N = number of spikes, n_d = spike feature dim
            r: size (N, n_c) array (normalized E_q(z)[z])  
            nu_k: size (n_k,) array (alternative rep of E_q(y)[y])
            mu: size (n_c, n_d) array  (GMM means)
        
        Returns:
            p: float; prob. of choosing 0 or 1 for binary variable. 
            mu: size (n_c, n_d) array (updated gmm means)
            cov: size (n_c, n_d, n_d) array (updated gmm covariance matrix)
        """
        
        p = nu_k.sum() / self.test_n_k
        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
                ',i,ip,id->pd', 1/norm[c], r[:,c], s-mu[c], s-mu[c]
               ) for c in range(self.n_c)]
        
        return p, mu, cov
    
    
    def encode(self, s, y, max_iter=20, eps=1e-6):
        """
        Run the encoder model.
        
        Args:
            s: size (N, n_d) array, N = number of spikes, n_d = spike feature dim
            y: size (N, n_p) array (convenient rep of observed y for einsum)
        
        Returns:
            r: size (N, n_c) array (updated normalized E_q(z)[z]) 
            lam: (n_c, n_t, 2) array (updated unnormalized lambda)
            mu: (n_c, n_d) array (updated GMM means) 
            cov: (n_c, n_d, n_d) array (updated GMM covariance matrix)
            elbos: a list of ELBOs
        """
        
        # initialize 
        s = torch.tensor(s)
        r = torch.ones((s.shape[0], self.n_c)) / self.n_c
        lam = self.init_lam.clone()
        mu, cov = self.init_mu.clone(), self.init_cov.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        
        ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=self.init_cov)
        elbo = self._compute_encoder_elbo(r, y, ll, norm_lam)
        elbos = [elbo]
        
        for i in tqdm(range(max_iter), desc="Train CAVI"):
            # E step
            r = self._encode_e_step(r, y, ll, norm_lam)
            # M step
            mu, cov, lam, norm_lam = self._encode_m_step(s, r, y, mu, lam)
            # compute elbo
            ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=self.init_cov)
            elbo = self._compute_encoder_elbo(r, y, ll, norm_lam)
            elbos.append(elbo)
            
        return r, lam, mu, torch.stack(cov), elbos
    
    
    def decode(self, s, init_p, init_mu, init_cov, init_lam, test_ks, test_ids, max_iter=20, eps=1e-6):
        """
        Run the decoder model.
        
        Args:
            s: size (N, n_d) array, N = number of spikes, n_d = spike feature dim
            init_p: float; initial prob. of choosing 0 or 1 for binary variable
            init_mu: size (n_c, n_d) array (initial GMM means)
            init_cov: size (n_c, n_d, n_d) array (initial GMM covariance matrix)
            init_lam: size (n_c, n_t, n_p) array (initial unnormalized lambda)
            test_ks: a list of arrays containing spike index 
            test_ids: test trial index
        
        Returns:
            r: size (n, c) array (updated normalized E_q(z)[z])
            nu_k: size (k,) array (updated E_q(y)[y])
            mu: size (c, d) array (updated GMM means)
            cov: size (c, d, d) array (updated GMM covariance matrix)
            p: float; estimated prob. of choosing 0 or 1 for binary variable
            elbos: a list of ELBOs. 
        """
        
        # initialize 
        s = torch.tensor(s)
        p = torch.tensor([init_p])
        r = torch.ones((s.shape[0], self.n_c)) / self.n_c
        mu, cov = init_mu.clone(), init_cov.clone()
        lam = init_lam.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        nu_k = torch.rand(self.test_n_k)
        nu = torch.zeros(s.shape[0])
        for k in range(self.test_n_k):
            nu[test_ks == test_ids[k]] = nu_k[k]
        nu = nu.reshape(-1,1)
        
        ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=init_cov)
        elbo = self._compute_decoder_elbo(r, ll, norm_lam, nu, nu_k, p)
        elbos = [elbo]
        
        for i in tqdm(range(max_iter), desc="Decode CAVI"):
            # E step
            r, nu, nu_k = self._decode_e_step(r, ll, norm_lam, nu, nu_k, p)
            # M step
            p, mu, cov = self._decode_m_step(s, r, nu_k, mu)
            # compute elbo
            ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=init_cov)
            elbo = self._compute_decoder_elbo(r, ll, norm_lam, nu, nu_k, p)
            elbos.append(elbo)
            
        return r, nu_k, mu, torch.stack(cov), p, elbos
    
    
    def eval_perf(self, nu_k, y_test):
        """
        Evaluate the decoder performance.
        
        Args:
            nu_k: size (k,) array (predicted prob. of choosing 1 choice)
            y_test: size (k,) array (observed y in test set)
        Returns:
            acc: float (accuracy)
            auc: float (AUC)
        """
        
        acc = accuracy_score(y_test, 1.*(nu_k>.5))
        auc = roc_auc_score(y_test, nu_k)
        print(f'accuracy: {acc:.3f}')
        print(f'auc: {auc:.3f}')
        
        return acc, auc

    
def compute_cavi_weight_matrix(
    x, 
    y_train, 
    y_pred, 
    train, 
    test, 
    post_params
):
    """
    Compute the posterior dynamic mixture weights for GMM and the posterior weight matrix 
    as input to the behavior decoder.

    Args:
        x: a nested list w/ the structure:
           for each k:
               for each t:
                   size (n_t_k, 1+n_d) array, n_d = spike feature dim
        y_train (y_pred): size (n_k,) or (n_k, n_t) array
        train: trial index in the train set
        test: trial index in the test set
        post_params: a dict of model parameters that contains b, beta, means and covs 

    Returns:
        mixture_weights: size (n_k, n_c, n_t) array
        weight_matrix: size (n_k, n_c, n_t) array
    """
    
    aligned_idxs = np.append(train, test)
    y_train, y_pred = y_train.squeeze(), y_pred.squeeze()
    y = np.hstack([y_train, y_pred]).astype(int)
    n_k = len(y) 
    n_c, n_t, _ = post_params["lambdas"].shape
    
    post_gmm = GaussianMixture(n_components=n_c, covariance_type='full')
    post_gmm.means_ = post_params["means"]
    post_gmm.covariances_ = post_params["covs"]
    post_gmm.precisions_cholesky_ = np.linalg.cholesky(
        np.linalg.inv(post_params["covs"])
    )
    
    mixture_weights = post_params["lambdas"] / post_params["lambdas"].sum(0)
    
    weight_matrix = np.zeros((n_k, n_c, n_t)) 
    for i in tqdm(range(n_k), desc="Compute weight matrix"):
        k = aligned_idxs[i]
        for t in range(n_t):
            post_gmm.weights_ = mixture_weights[:,t,y[k]]
            if len(x[k][t]) > 0:
                weight_matrix[k,:,t] = post_gmm.predict_proba(x[k][t][:,1:]).sum(0)
                
    match_idxs = [np.argwhere(np.array(aligned_idxs) == k).item() for k in range(n_k)]
    weight_matrix = weight_matrix[match_idxs]

    return mixture_weights, weight_matrix
    
    
def compute_lambda_for_cavi(
    bin_spike_features,
    bin_behaviors,
    gmm, 
    n_p = 2
):
    """
    Compute lambda from the observed data to initialize CAVI.

    Args:
        bin_spike_features: a nested list w/ the structure:
                            for each k:
                                for each t:
                                    size (n_t_k, 1+n_d) array, n_d = spike feature dim
        bin_behaviors: size (n_k,) array
        gmm: an object from sklearn.mixture.GaussianMixture()
        n_p: number of categories of the behavior variable

    Returns:
        lambdas: size (n_c, n_t, n_p) array, 
                 n_c = number of Gaussian mixture components, 
                 n_t = number of time bins
        p: prob. of choosing 0 or 1 for the binary variable
    """

    n_c = gmm.means_.shape[0]
    n_k = len(bin_spike_features)
    n_t = len(bin_spike_features[0])
    
    lambdas = []
    for k in tqdm(range(n_k), desc="Initialize variational params"):
        lambdas_per_trial = []
        for t in range(n_t):
            lamdas_per_time_bin = np.zeros((n_c, n_p))
            spike_features_per_time_bin = bin_spike_features[k][t][:,1:]
            if len(spike_features_per_time_bin) > 0:
                spike_labels = gmm.predict(spike_features_per_time_bin)
                for c in range(n_c):
                    if bin_behaviors[k] == 0:
                        lamdas_per_time_bin[c, 0] = np.sum(spike_labels == c)
                    else:
                        lamdas_per_time_bin[c, 1] = np.sum(spike_labels == c)
            lambdas_per_trial.append(lamdas_per_time_bin)
        lambdas.append(lambdas_per_trial)

    n_left = np.sum(bin_behaviors == 0)
    n_right = np.sum(bin_behaviors == 1)
    p = n_right / (n_right + n_left)
    lambdas = ( np.array(lambdas).sum(0) / np.array([n_left, n_right]) ).transpose(1,0,2)
    
    return lambdas, p
    
    
    
    
    