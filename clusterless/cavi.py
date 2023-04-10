import numpy as np
import torch
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, roc_auc_score
from clusterless.viz import plot_decoder_input


def safe_log(x, minval=1e-10):
    return torch.log(x + minval)

def safe_divide(x, y):
    return torch.clip(x / y, min = 0, max = 1)

class CAVI():
    def __init__(
        self, 
        init_mu, 
        init_cov, 
        init_lam, 
        train_ks, 
        train_ts, 
        test_ks, 
        test_ts
    ):
        '''
        CAVI can only be used for the binary choice variable. 
        
        Inputs:
        -------
        init_mu:  (c, d) array; c = # of gmm components, d = spike feature dimension. 
        init_cov: (c, d, d) array.
        init_lam: (c, t, 2) array; t = # of time bins. 
        train_ks: a list of arrays containing spike index in each training trial. 
        train_ts: a list of arrays containing spike index in each training time bin.
        test_ks: a list of arrays containing spike index in each test trial. 
        test_ts: a list of arrays containing spike index in each test time bin.
        '''
        
        self.train_n_k = len(train_ks)
        self.test_n_k = len(test_ks)
        self.n_t = len(train_ts)
        self.n_c = init_mu.shape[0]
        self.n_d = init_mu.shape[1]
        self.init_mu = torch.tensor(init_mu)
        self.init_cov = torch.tensor(init_cov)
        self.init_lam = torch.tensor(init_lam)
        self.train_ks = train_ks
        self.train_ts = train_ts
        self.test_ks = test_ks
        self.test_ts = test_ts
        
    def _compute_gmm_log_pdf(self, s, mu, cov, safe_cov=None):
        '''
        Inputs:
        -------
        s: (n, d) array; n = # of all spikes in the recording, d = spike feature dimension.
        mu: (c, d) array; updated gmm means. 
        cov: (c, d, d) array; updated gmm covariance matrix. 
        safe_cov: (c, d, d) array; alternative covariance matrix to use in case cov is not PSD.
         
        Outputs:
        -------
        ll: (n, c) array; computed log-likelihood. 
        '''
        ll = []
        for j in range(self.n_c):
            try:
                ll.append(
                    torch.tensor(
                        multivariate_normal.logpdf(s, mu[j], cov[j])
                    )
                )
            except np.linalg.LinAlgError:
                # This error occurs when the covariance matrix is not PSD.
                # We can use the initial covariance matrix as a replacement to ensure numerical stability.
                # TO DO: Need a better solution. 
                ll.append(
                    torch.tensor(
                        multivariate_normal.logpdf(s, mu[j], safe_cov[j])
                    )
                )
                print(f'Covariance matrix of component {j} is not PSD.')
        return torch.vstack(ll).T 
    
    
    def _compute_encoder_elbo(self, r, y, ll, norm_lam):
        '''
        Inputs:
        -------
        r:  (n, c) array; normalized E_q(z)[z].  
        y:  (n, 2) array; a convenient representation of the observed y for einsum. 
        ll: (n, c) array; computed log-likelihood. 
        norm_lam: (c, t, 2) array; normalized lambda. 
        
        Outputs:
        -------
        elbo: Encoder ELBO.
        '''
        elbo = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,j], ll[:,j]) for j in range(self.n_c)]
        ))
        elbo += torch.tensor(
            [ torch.einsum('ij,il,j->', r[self.train_ts[t]], 
                           y[self.train_ts[t]], norm_lam[:,t,1]) +
              torch.einsum('ij,il,j->', r[self.train_ts[t]], 
                           1-y[self.train_ts[t]], norm_lam[:,t,0]) for t in range(self.n_t) ]
            ).sum()
        elbo -= torch.einsum('ij,ij->', safe_log(r), r)
        return elbo
    
    
    def _compute_decoder_elbo(self, r, ll, norm_lam, nu, nu_k, p):
        '''
        Inputs:
        -------
        r:  (n, c) array; normalized E_q(z)[z].  
        ll: (n, c) array; computed log-likelihood. 
        norm_lam: (c, t, 2) array; normalized lambda. 
        nu: (n, 2) array; a convenient representation of E_q(y)[y] for einsum. 
        nu_k: (k,) array; alternative representation of E_q(y)[y].
        p: float; probability of choosing left or right in a visual decision task. 
        
        Outputs:
        -------
        elbo: Decoder ELBO.
        '''
        elbo = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,j], ll[:,j]) for j in range(self.n_c)]
        ))
        elbo += torch.tensor(
            [ torch.einsum('ij,il,j->', r[self.test_ts[t]], 
                           nu[self.test_ts[t]], norm_lam[:,t,1]) +
              torch.einsum('ij,il,j->', r[self.test_ts[t]], 
                           1-nu[self.test_ts[t]], norm_lam[:,t,0]) for t in range(self.n_t) ]
            ).sum()
        elbo += torch.sum(nu_k * safe_log(p) + (1-nu_k) * safe_log(1-p))
        elbo -= torch.einsum('ij,ij->', safe_log(r), r)
        elbo -= torch.sum(safe_log(nu_k) * nu_k)
        return elbo
    
    
    def _encode_e_step(self, r, y, ll, norm_lam):
        '''
        Inputs:
        -------
        r:  (n, c) array; normalized E_q(z)[z].  
        y:  (n, 2) array; a convenient representation of the observed y for einsum. 
        ll: (n, c) array; computed log-likelihood. 
        norm_lam: (c, t, 2) array; normalized lambda. 
        
        Outputs:
        -------
        r:  (n, c) array; updated normalized E_q(z)[z].  
        '''
        for t in range(self.n_t):
            r[self.train_ts[t]] = torch.exp( ll[self.train_ts[t]] + \
                      torch.einsum('il,j->ij', y[self.train_ts[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-y[self.train_ts[t]], norm_lam[:,t,0])
            )
            r[self.train_ts[t]] = torch.einsum('ij,i->ij', r[self.train_ts[t]], 1/r[self.train_ts[t]].sum(1))
        return r
        
    
    def _encode_m_step(self, s, r, y, mu, lam):
        '''
        Inputs:
        -------
        s:   (n, d) array; n = # of all spikes in the recording, d = spike feature dimension.
        r:   (n, c) array; normalized E_q(z)[z].  
        y:   (n, 2) array; a convenient representation of the observed y for einsum. 
        mu:  (c, d) array; gmm means.
        lam: (c, t, 2) array; unnormalized lambda.
        
        Outputs:
        -------
        mu:  (c, d) array; updated gmm means. 
        cov: (c, d, d) array; updated gmm covariance matrix. 
        lam: (c, t, 2) array; updated unnormalized lambda.
        norm_lam: (c, t, 2) array; updated normalized lambda.
        '''
        for j in range(self.n_c):
            no_j_idx = torch.cat([torch.arange(j), torch.arange(j+1, self.n_c)])
            lam_sum_no_j = lam[no_j_idx,:,:].sum(0)
            for t in range(self.n_t):
                num1 = torch.einsum('i,il,->', r[self.train_ts[t],j], y[self.train_ts[t]], lam_sum_no_j[t,1])
                denom1 = np.einsum('ij,il->', r[self.train_ts[t]][:,no_j_idx], y[self.train_ts[t]])
                num0 = torch.einsum('i,il,->', r[self.train_ts[t],j], 1-y[self.train_ts[t]], lam_sum_no_j[t,0])
                denom0 = np.einsum('ij,il->', r[self.train_ts[t]][:,no_j_idx], 1-y[self.train_ts[t]])
                lam[j,t,1], lam[j,t,0] = num1 / denom1, num0 / denom0
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
                    ',i,ip,id->pd', 1/norm[j], r[:,j], s-mu[j], s-mu[j]) for j in range(self.n_c)]
        return mu, cov, lam, norm_lam
    
    
    def _decode_e_step(self, r, ll, norm_lam, nu, nu_k, p):
        '''
        Inputs:
        -------
        r:   (n, c) array; normalized E_q(z)[z].  
        ll: (n, c) array; computed log-likelihood. 
        norm_lam: (c, t, 2) array; normalized lambda.
        nu: (n, 2) array; a convenient representation of E_q(y)[y] for einsum. 
        nu_k: (k,) array; alternative representation of E_q(y)[y].
        p: float; probability of choosing left or right in a visual decision task. 
        
        Outputs:
        -------
        r:   (n, c) array; updated normalized E_q(z)[z].  
        nu: (n, 2) array; updated E_q(y)[y] for einsum. 
        nu_k: (k,) array; updated E_q(y)[y].
        '''
        for t in range(self.n_t):
            r[self.test_ts[t]] = torch.exp( ll[self.test_ts[t]] + \
                      torch.einsum('il,j->ij', nu[self.test_ts[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-nu[self.test_ts[t]], norm_lam[:,t,0])
            )
            r[self.test_ts[t]] = torch.einsum('ij,i->ij', r[self.test_ts[t]], 1/r[self.test_ts[t]].sum(1))
        for k in range(self.test_n_k):
            y_tilde0, y_tilde1 = safe_log(1-p), safe_log(p)
            for t in range(self.n_t):
                k_t_idx = np.intersect1d(self.test_ks[k], self.test_ts[t])
                y_tilde0 += torch.einsum('ij,j->', r[k_t_idx], norm_lam[:,t,0])
                y_tilde1 += torch.einsum('ij,j->', r[k_t_idx], norm_lam[:,t,1])
            # exp(y_tilde) explodes to 0 so need to offset to ensure numerical stability
            # TO DO: Need a better solution. 
            offset = 1. / (torch.min(torch.tensor([y_tilde0, y_tilde1])) / -745.) 
            y_tilde0, y_tilde1 = torch.exp(y_tilde0 * offset), torch.exp(y_tilde1 * offset)
            nu_k[k] = safe_divide(y_tilde1, y_tilde0+y_tilde1)
            nu[self.test_ks[k]] = nu_k[k]
        return r, nu, nu_k
    
    
    def _decode_m_step(self, s, r, nu_k, mu):
        '''
        Inputs:
        -------
        s:   (n, d) array; n = # of all spikes in the recording, d = spike feature dimension.
        r:   (n, c) array; normalized E_q(z)[z].  
        nu_k: (k,) array; alternative representation of E_q(y)[y].
        mu:  (c, d) array; gmm means.
        
        Outputs:
        -------
        p: float; probability of choosing left or right in a visual decision task. 
        mu:  (c, d) array; updated gmm means. 
        cov: (c, d, d) array; updated gmm covariance matrix. 
        '''
        p = nu_k.sum() / self.test_n_k
        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
                    ',i,ip,id->pd', 1/norm[j], r[:,j], s-mu[j], s-mu[j]) for j in range(self.n_c)]
        return p, mu, cov
    
    
    def encode(self, s, y, max_iter=20, eps=1e-6):
        '''
        Inputs:
        -------
        s:   (n, d) array; n = # of all spikes in the recording, d = spike feature dimension.
        y:   (n, 2) array; a convenient representation of the observed y for einsum. 
        
        Outputs:
        -------
        r:   (n, c) array; updated normalized E_q(z)[z]. 
        lam: (c, t, 2) array; updated unnormalized lambda.
        mu:  (c, d) array; updated gmm means. 
        cov: (c, d, d) array; updated gmm covariance matrix. 
        elbos: a list of encoder ELBOs. 
        '''
        # initialize 
        s = torch.tensor(s)
        r = torch.ones((s.shape[0], self.n_c)) / self.n_c
        lam = self.init_lam.clone()
        mu, cov = self.init_mu.clone(), self.init_cov.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=self.init_cov)
        elbo = self._compute_encoder_elbo(r, y, ll, norm_lam)
        convergence = 1.
        elbos = [elbo]
        print(f'initial elbo: {elbos[-1]:.2f}')
        
        it = 1
        while convergence > eps or convergence < 0: 
            # E step
            r = self._encode_e_step(r, y, ll, norm_lam)
            # M step
            mu, cov, lam, norm_lam = self._encode_m_step(s, r, y, mu, lam)
            # compute elbo
            ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=self.init_cov)
            elbo = self._compute_encoder_elbo(r, y, ll, norm_lam)
            elbos.append(elbo)
            convergence = elbos[-1] - elbos[-2]
            print(f'iter: {it} elbo: {elbos[-1]:.2f}.')
            it +=1 
            if it > max_iter: 
                print('reached max iter allowed.')
                break
        if abs(convergence) <= eps:
            print('converged.')
        return r, lam, mu, torch.stack(cov), elbos
    
    
    def decode(self, s, init_p, init_mu, init_cov, init_lam, test_ks, test_ids, max_iter=20, eps=1e-6):
        '''
        Inputs:
        -------
        s:   (n, d) array; n = # of all spikes in the recording, d = spike feature dimension.
        init_p: float; initial prob. of choosing left or right in a visual decision task 
                (from encoder or observed data).
        init_mu: (c, d) array; initial gmm means (from encoder or observed data). 
        init_cov: (c, d, d) array; initial gmm covariance matrix (from encoder or observed data). 
        init_lam: (c, t, 2) array; initial unnormalized lambda (from encoder or observed data).
        test_ks: a list of arrays containing spike index in each test trial. 
        test_ids: test trial indices. 
        
        Outputs:
        -------
        r:   (n, c) array; updated normalized E_q(z)[z]. 
        nu_k: (k,) array; updated E_q(y)[y].
        mu:  (c, d) array; updated gmm means. 
        cov: (c, d, d) array; updated gmm covariance matrix. 
        p: float; estimated prob. of choosing left or right in a visual decision task.
        elbos: a list of decoder ELBOs. 
        '''
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
        convergence = 1.
        elbos = [elbo]
        print(f'initial elbo: {elbos[-1]:.2f}')
        
        it = 1
        while convergence > eps or convergence < 0:
            # E step
            r, nu, nu_k = self._decode_e_step(r, ll, norm_lam, nu, nu_k, p)
            # M step
            p, mu, cov = self._decode_m_step(s, r, nu_k, mu)
            # compute elbo
            ll = self._compute_gmm_log_pdf(s, mu, cov, safe_cov=init_cov)
            elbo = self._compute_decoder_elbo(r, ll, norm_lam, nu, nu_k, p)
            elbos.append(elbo)
            convergence = elbos[-1] - elbos[-2]
            print(f'iter: {it} elbo: {elbos[-1]:.2f}.')
            it +=1 
            if it > max_iter: 
                print('reached max iter allowed.')
                break
        if abs(convergence) <= eps:
            print('converged.')
        return r, nu_k, mu, torch.stack(cov), p, elbos
    
    
    def eval_perf(self, nu_k, y_test):
        acc = accuracy_score(y_test, 1.*(nu_k>.5))
        auc = roc_auc_score(y_test, nu_k)
        print(f'accuracy: {acc:.3f}')
        print(f'auc: {auc:.3f}')
        return acc, auc
    
    
def encode_gmm(data, lams, means, covs, train, test, y_train, y_pred):
    '''
    Encoding the gmm with the dynamic mixing propotions. 

    Inputs:
    -------
    data: a nested list of spike features; data[k][t] contains spikes that fall into k-th trial and
          t-th time bin. data[k][t] is a (spike channels, spike features) array.
    train: training trial indices. 
    test:  test trial indices. 
    lams: updated lambda (c,t,2) from either the encoder or decoder.
    means: updated gmm means (c,d) from either the encoder or decoder.
    covs: updated gmm covariances (c,d,d) from either the encoder or decoder.
    y_train: discrete or continuous behaviors in the training trials. 
    y_pred:  initially predicted behaviors in the test trials; can obtain using either
             multi-unit thresholding or vanilla gmm (with fixed mixing proportions).

    Outputs:
    -------
    encoded_pis: dynamic mixing proportions; (k, c, t) array. 
    encoded_weights: posterior assignment weight matrix from the encoded gmm; (k, c, t) array. 
    '''
    trial_idx = np.append(train, test)
    y = np.vstack([y_train, y_pred])
    n_k = len(trial_idx) 
    n_c, n_t = lams.shape[0], lams.shape[1]
    
    encoded_pis = lams / lams.sum(0)
    encoded_weights = np.zeros((n_k, n_c, n_t))
    for i, k in enumerate(trial_idx):
        for t in range(n_t):
            encoded_gmm = GaussianMixture(n_components=n_c, covariance_type='full')
            encoded_gmm.weights_ = encoded_pis[:, t, y[i]]
            encoded_gmm.means_ = means
            encoded_gmm.covariances_ = covs
            encoded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
            if len(data[k][t]) > 0:
                encoded_weights[k,:,t] = encoded_gmm.predict_proba(data[k][t][:,1:]).sum(0)

    return encoded_pis, encoded_weights

