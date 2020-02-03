import numpy as np
from scipy.stats import norm



class KG_optimizer():

    def __init__(self,model):
        """

        :param model: GPy trained model
        """
        self.X = model.X
        self.model = model

    def __call__(self,x_space):
        """
        receives vector of points to calculate Knowledge Gradient at each one of the locations
        :param x_space: vector of input locations
        :return: kg value for each location
        """
        mu_data, var_data = self.model.predict(self.X)
        mu_data = mu_data.reshape(-1)

        kg = []
        for i in range(len(x_space)):
            # just adapting dimensions
            x_new = x_space[i]
            x_new = np.array([x_new])
            X_tilde = np.concatenate((x_new, self.X))
            # calculating posterior vector covariance in new position. the posterior covariance includes correlation new point
            # with all previous sampled points.
            mu_xnew = self.model.predict(x_new)[0]
            S_xnew = self.model.posterior_covariance_between_points(x_new, X_tilde)

            var_xnew = self.model.predict(x_new)[1]  # includes likelihood variance

            SS = S_xnew / np.sqrt(var_xnew)
            MM = np.concatenate((mu_xnew.reshape(-1), mu_data.reshape(-1)))
            # Finally compute KG!

            out = self.KG(MM.reshape(-1), SS.reshape(-1))
            kg.append(out)
        return kg

    def KG(self, mu, sig):
        """
        Takes a set of intercepts and gradients of linear functions and returns
        the average hieght of the max of functions over Gaussain input.

        ARGS
            mu: length n vector, initercepts of linear functions
            sig: length n vector, gradients of linear functions

        RETURNS
            out: scalar value is gaussain expectation of epigraph of lin. funs
        """

        n = len(mu)
        O = sig.argsort()
        a = mu[O]
        b = sig[O]
        A = [0]
        C = [-float("inf")]
        while A[-1] < n - 1:
            s = A[-1]
            si = range(s + 1, n)
            Ci = -(a[s] - a[si]) / (b[s] - b[si])
            bestsi = np.argmin(Ci)
            C.append(Ci[bestsi])
            A.append(si[bestsi])

        C.append(float("inf"))
        cdf_C = norm.cdf(C)
        diff_CDF = cdf_C[1:] - cdf_C[:-1]
        pdf_C = norm.pdf(C)
        diff_PDF = pdf_C[1:] - pdf_C[:-1]

        out = np.sum(a[A] * diff_CDF - b[A] * diff_PDF) #- np.max(mu)

        return out


