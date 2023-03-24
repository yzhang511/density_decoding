import numpy as np
import random
import torch
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, roc_auc_score

# helper functions
def safe_log(x, minval=1e-10):
    return torch.log(x + minval)

def safe_divide(x, y):
    return torch.clip(x / y, min = 0, max = 1)


class CAVI():
    def __init__(self, init_mu, init_cov, init_lam, 
                 train_k_ids, train_t_ids, test_k_ids, test_t_ids):
        
        self.train_K = len(train_k_ids)
        self.test_K = len(test_k_ids)
        self.T = len(train_t_ids)
        self.C = init_mu.shape[0]
        self.D = init_mu.shape[1]
        self.init_mu = init_mu
        self.init_cov = init_cov
        self.init_lam = init_lam
        self.train_k_ids = train_k_ids
        self.train_t_ids = train_t_ids
        self.test_k_ids = test_k_ids
        self.test_t_ids = test_t_ids
        
    
    def _compute_normal_log_dens(self, s, mu, cov):
        
        log_dens = []
        for j in range(self.C):
            log_dens.append(
                torch.tensor(
                    multivariate_normal.logpdf(s, mu[j], cov[j])
                )
            )
        return torch.vstack(log_dens).T # (*, C)
    
    
    def _compute_enc_elbo(self, r, y, log_dens, norm_lam):
        
        elbo_1 = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,j], log_dens[:,j]) for j in range(self.C)]
        ))

        elbo_2 = torch.tensor(
            [ torch.einsum('ij,il,j->', r[self.train_t_ids[t]], 
                           y[self.train_t_ids[t]], norm_lam[:,t,1]) +
              torch.einsum('ij,il,j->', r[self.train_t_ids[t]], 
                           1-y[self.train_t_ids[t]], norm_lam[:,t,0]) for t in range(self.T) ]
            ).sum()

        elbo_3 = - torch.einsum('ij,ij->', safe_log(r), r)

        elbo = elbo_1 + elbo_2 + elbo_3 

        return elbo
    
    
    def _compute_dec_elbo(self, r, log_dens, norm_lam, nu, nu_k, p):
        
        elbo_1 = torch.sum(torch.tensor(
            [torch.einsum('i,i->', r[:,j], log_dens[:,j]) for j in range(self.C)]
        ))

        elbo_2 = torch.tensor(
            [ torch.einsum('ij,il,j->', r[self.test_t_ids[t]], 
                           nu[self.test_t_ids[t]], norm_lam[:,t,1]) +
              torch.einsum('ij,il,j->', r[self.test_t_ids[t]], 
                           1-nu[self.test_t_ids[t]], norm_lam[:,t,0]) for t in range(self.T) ]
            ).sum()

        elbo_3 = torch.sum(nu_k * safe_log(p) + (1-nu_k) * safe_log(1-p))

        elbo_4 = - torch.einsum('ij,ij->', safe_log(r), r)

        elbo_5 = - torch.sum(safe_log(nu_k) * nu_k)

        elbo = elbo_1 + elbo_2 + elbo_3 + elbo_4 + elbo_5

        return elbo
    
    
    def _encode_e_step(self, r, y, log_dens, norm_lam):
        
        for t in range(self.T):
            r[self.train_t_ids[t]] = torch.exp( log_dens[self.train_t_ids[t]] + \
                      torch.einsum('il,j->ij', y[self.train_t_ids[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-y[self.train_t_ids[t]], norm_lam[:,t,0])
            )
            r[self.train_t_ids[t]] = torch.einsum('ij,i->ij', r[self.train_t_ids[t]], 1/r[self.train_t_ids[t]].sum(1))
        return r
        
    
    def _encode_m_step(self, s, r, y, mu, lam):
        
        for j in range(self.C):
            no_j_idx = torch.cat([torch.arange(j), torch.arange(j+1, self.C)])
            lam_sum_no_j = lam[no_j_idx,:,:].sum(0)
            for t in range(self.T):
                num1 = torch.einsum('i,il,->', r[self.train_t_ids[t],j], y[self.train_t_ids[t]], lam_sum_no_j[t,1])
                denom1 = np.einsum('ij,il->', r[self.train_t_ids[t]][:,no_j_idx], y[self.train_t_ids[t]])
                num0 = torch.einsum('i,il,->', r[self.train_t_ids[t],j], 1-y[self.train_t_ids[t]], lam_sum_no_j[t,0])
                denom0 = np.einsum('ij,il->', r[self.train_t_ids[t]][:,no_j_idx], 1-y[self.train_t_ids[t]])
                lam[j,t,1], lam[j,t,0] = num1 / denom1, num0 / denom0
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))

        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
            ',i,ip,id->pd', 1/norm[j], r[:,j], s-mu[j], s-mu[j] ) for j in range(self.C)]
        
        return mu, cov, lam, norm_lam
    
    
    def _decode_e_step(self, r, log_dens, norm_lam, nu, nu_k, p):
        
        for t in range(self.T):
            r[self.test_t_ids[t]] = torch.exp( log_dens[self.test_t_ids[t]] + \
                      torch.einsum('il,j->ij', nu[self.test_t_ids[t]], norm_lam[:,t,1]) + \
                      torch.einsum('il,j->ij', 1-nu[self.test_t_ids[t]], norm_lam[:,t,0])
            )
            r[self.test_t_ids[t]] = torch.einsum('ij,i->ij', r[self.test_t_ids[t]], 1/r[self.test_t_ids[t]].sum(1))
        
        for k in range(self.test_K):
            y_tilde0, y_tilde1 = safe_log(1-p), safe_log(p)
            for t in range(self.T):
                k_t_ids = np.intersect1d(self.test_k_ids[k], self.test_t_ids[t])
                y_tilde0 += torch.einsum('ij,j->', r[k_t_ids], norm_lam[:,t,0])
                y_tilde1 += torch.einsum('ij,j->', r[k_t_ids], norm_lam[:,t,1])
            # y_tilde explode to 0 after exp(); need offset to ensure y_tilde stay in range
            offset = 1. / (torch.min(torch.tensor([y_tilde0, y_tilde1])) / -745.) 
            y_tilde0, y_tilde1 = torch.exp(y_tilde0 * offset), torch.exp(y_tilde1 * offset)
            nu_k[k] = safe_divide(y_tilde1, y_tilde0 + y_tilde1)
            nu[self.test_k_ids[k]] = nu_k[k]
            
        return r, nu, nu_k
    
    
    def _decode_m_step(self, s, r, nu_k, mu):
        
        p = nu_k.sum() / self.test_K
    
        norm = r.sum(0)
        mu = torch.einsum('j,ij,ip->jp', 1/norm, r, s)
        cov = [torch.einsum(
            ',i,ip,id->pd', 1/norm[j], r[:,j], s-mu[j], s-mu[j]) for j in range(self.C)]
        
        return p, mu, cov
    
    
    def encode(self, s, y, max_iter=20, eps=1e-6):
        # initialize
        r = torch.ones((s.shape[0], self.C)) / self.C
        lam = self.init_lam.clone()
        mu, cov = self.init_mu.clone(), self.init_cov.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        
        # compute initial elbo
        log_dens = self._compute_normal_log_dens(s, mu, cov)
        elbo = self._compute_enc_elbo(r, y, log_dens, norm_lam)
        convergence = 1.
        elbos = [elbo]
        print(f'initial elbo: {elbos[-1]:.2f}')
        
        it = 1
        while convergence > eps or convergence < 0: 
            # E step
            r = self._encode_e_step(r, y, log_dens, norm_lam)
            
            # M step
            mu, cov, lam, norm_lam = self._encode_m_step(s, r, y, mu, lam)
            
            # compute new elbo
            log_dens = self._compute_normal_log_dens(s, mu, cov)
            elbo = self._compute_enc_elbo(r, y, log_dens, norm_lam)
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
    
    
    def decode(self, s, init_p, init_mu, init_cov, init_lam, 
                test_k_ids, test_ids, max_iter=20, eps=1e-6):
        # initialize
        p = init_p.clone()
        r = torch.ones((s.shape[0], self.C)) / self.C
        mu, cov = init_mu.clone(), init_cov.clone()
        lam = init_lam.clone()
        norm_lam = safe_log(lam) - safe_log(lam.sum(0))
        nu_k = torch.rand(self.test_K)
        nu = torch.zeros(s.shape[0])
        for k in range(self.test_K):
            nu[test_k_ids == test_ids[k]] = nu_k[k]
        nu = nu.reshape(-1,1)
        
        # compute initial elbo
        log_dens = self._compute_normal_log_dens(s, mu, cov)
        elbo = self._compute_dec_elbo(r, log_dens, norm_lam, nu, nu_k, p)
        convergence = 1.
        elbos = [elbo]
        print(f'initial elbo: {elbos[-1]:.2f}')
        
        it = 1
        while convergence > eps or convergence < 0:
            # E step
            r, nu, nu_k = self._decode_e_step(r, log_dens, norm_lam, nu, nu_k, p)
            
            # M step
            p, mu, cov = self._decode_m_step(s, r, nu_k, mu)
            
            # compute new elbo
            log_dens = self._compute_normal_log_dens(s, mu, cov)
            elbo = self._compute_dec_elbo(r, log_dens, norm_lam, nu, nu_k, p)
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
    
    
    def eval_performance(self, nu_k, y_test):
        acc = accuracy_score(y_test, 1. * ( nu_k > .5 ))
        auc = roc_auc_score(y_test, nu_k)
        print(f'decoding accuracy is {acc:.2f}')
        print(f'decoding auc is {auc:.2f}')
        return acc, auc
    
    
    
    def encode_gmm(self, trials, lams, means, covs, train, test, y_train, y_hat):
        
        enc_pis = lams / lams.sum(0)
        
        enc_all = np.zeros((len(train) + len(test), self.C, self.T))
        for i, k in enumerate(train):
            for t in range(self.T):
                enc_gmm = GaussianMixture(n_components=self.C, covariance_type='full')
                enc_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
                enc_gmm.weights_ = enc_pis[:,t,y_train[i]]
                enc_gmm.means_ = means
                enc_gmm.covariances_ = covs
                if len(trials[k][t]) > 0:
                    enc_all[k,:,t] = enc_gmm.predict_proba(trials[k][t][:,1:]).sum(0)
                    
        for i, k in enumerate(test):
            for t in range(self.T):
                enc_gmm = GaussianMixture(n_components=self.C, covariance_type='full')
                enc_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covs))
                enc_gmm.weights_ = enc_pis[:,t,y_hat[i]]
                enc_gmm.means_ = means
                enc_gmm.covariances_ = covs
                if len(trials[k][t]) > 0:
                    enc_all[k,:,t] = enc_gmm.predict_proba(trials[k][t][:,1:]).sum(0)
        
        return enc_pis, enc_all
        
