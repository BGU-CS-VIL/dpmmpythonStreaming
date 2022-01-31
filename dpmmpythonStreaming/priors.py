import julia
from julia import DPMMSubClustersStreaming
import numpy as np

class prior:
    def to_julia_prior(self):
        pass

class niw(prior):
    def __init__(self, kappa, mu, nu, psi):
        if nu < len(mu):
            raise Exception('nu should be atleast the Dim')
        self.kappa = kappa
        self.mu = mu
        self.nu = nu
        self.psi = psi
        

    def to_julia_prior(self):
        return DPMMSubClustersStreaming.niw_hyperparams(self.kappa,self.mu,self.nu, self.psi)


class multinomial(prior):
    def __init__(self, alpha,dim = 1):
        if isinstance(alpha,np.ndarray):
            self.alpha = alpha
        else:
            self.alpha = np.ones(dim)*alpha
        
        

    def to_julia_prior(self):
        return DPMMSubClustersStreaming.multinomial_hyper(self.alpha)

class compact_multinomial(prior):
    def __init__(self, alpha):
        if isinstance(alpha,np.ndarray):
            self.alpha = alpha
        else:
            self.alpha = np.ones(dim)*alpha       
        

    def to_julia_prior(self):
        return DPMMSubClustersStreaming.compact_mnm_hyper(self.alpha)

