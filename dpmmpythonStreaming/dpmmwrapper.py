import julia
julia.install()
from dpmmpythonStreaming.priors import niw, multinomial
from julia import DPMMSubClustersStreaming
import numpy as np



class DPMMPython:
    """
     Wrapper for the DPMMSubCluster Julia package
     """

    @staticmethod
    def create_prior(dim,mean_prior,mean_str,cov_prior,cov_str):
        """
        Creates a gaussian prior, if cov_prior is a scalar, then creates an isotropic prior scaled to that, if its a matrix
        uses it as covariance
        :param dim: data dimension
        :param mean_prior: if a scalar, will create a vector scaled to that, if its a vector then use it as the prior mean
        :param mean_str: prior mean psuedo count
        :param cov_prior: if a scalar, will create an isotropic covariance scaled to cov_prior, if a matrix will use it as
        the covariance.
        :param cov_str: prior covariance psuedo counts
        :return: DPMMSubClustersStreaming.niw_hyperparams prior
        """
        if isinstance(mean_prior,(int,float)):
            prior_mean = np.ones(dim) * mean_prior
        else:
            prior_mean = mean_prior

        if isinstance(cov_prior, (int, float)):
            prior_covariance = np.eye(dim) * cov_prior
        else:
            prior_covariance = cov_prior
        prior =niw(mean_str,prior_mean,dim + cov_str, prior_covariance)
        return prior

    @staticmethod
    def fit_init(data,alpha, prior,
            iterations= 100,init_clusters=1, verbose = False,
            burnout = 15, gt = None, epsilon = 0.1):
        """
        Wrapper for DPMMSubClustersStreaming fit, reffer to "https://bgu-cs-vil.github.io/DPMMSubClustersStreaming.jl/stable/usage/" for specification
        Note that directly working with the returned clusters can be problematic software displaying the workspace (such as PyCharm debugger).
        :return: labels, clusters, sublabels
        """

        results = DPMMSubClustersStreaming.dp_parallel_streaming(data, prior.to_julia_prior(), alpha, iterations,init_clusters,
                                        None, verbose, False,
                                        burnout,gt, epsilon)
        return results

    @staticmethod
    def fit_partial(model,iterations, t, data):
        DPMMSubClustersStreaming.run_model_streaming(model,iterations, t, data)
        return model

    @staticmethod
    def get_labels(model):
        return DPMMSubClustersStreaming.get_labels(model)

    @staticmethod
    def predict(model,data):
        return DPMMSubClustersStreaming.predict(model,data)


    
    @staticmethod
    def generate_gaussian_data(sample_count,dim,components,var):
        '''
        Wrapper for DPMMSubClustersStreaming cluster statistics
        :param sample_count: how much of samples
        :param dim: samples dimension
        :param components: number of components
        :param var: variance between componenets means
        :return: (data, gt)
        '''
        data = DPMMSubClustersStreaming.generate_gaussian_data(sample_count, dim, components, var)
        gt =  data[1]
        data = data[0]
        return data,gt

    @staticmethod
    def add_procs(procs_count):
        j = julia.Julia(compiled_modules=False)
        j.eval('using Distributed')
        j.eval('addprocs(' + str(procs_count) + ')')
        j.eval('@everywhere using DPMMSubClustersStreaming')


if __name__ == "__main__":
    j = julia.Julia(compiled_modules=False)
    data,gt = DPMMPython.generate_gaussian_data(10000, 2, 10, 100.0)
    prior = DPMMPython.create_prior(2, 0, 1, 1, 1)
    model= DPMMPython.fit_init(data,100.0,prior = prior,verbose = True, burnout = 5, gt = None, epsilon = 1.0)
    labels = DPMMPython.get_labels(model)
    print(labels)


