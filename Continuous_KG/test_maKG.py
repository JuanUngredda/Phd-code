import numpy as np
import GPyOpt

import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maKG import maKG

from parameter_distribution import ParameterDistribution
from utility import Utility
import matplotlib.pyplot as plt

# --- Function to optimize

class GP_test():
    """
A toy function GP

ARGS
 min: scalar defining min range of inputs
 max: scalar defining max range of inputs
 seed: int, RNG seed
 x_dim: designs dimension
 a_dim: input dimensions
 xa: n*d matrix, points in space to eval testfun
 NoiseSD: additive gaussaint noise SD

RETURNS
 output: vector of length nrow(xa)
 """

    def __init__(self, xamin, xamax, seed=11, x_dim=1):
        self.seed = seed
        self.dx = x_dim
        self.da = 0
        self.dxa = x_dim
        self.xmin = np.array([xamin for i in range(self.dxa)])
        self.xmax = np.array([xamax for i in range(self.dxa)])
        vr = 4.
        ls = 10
        self.HP =  [vr,ls]
        self.KERNEL = GPy.kern.RBF(input_dim=self.dxa, variance=vr, lengthscale=([ls] * self.dxa), ARD=True)
        self.generate_function()

    def __call__(self, xa, noise_std=0.1):
        assert len(xa.shape) == 2, "xa must be an N*d matrix, each row a d point"
        assert xa.shape[1] == self.dxa, "Test_func: wrong dimension inputed"

        xa = self.check_input(xa)

        ks = self.KERNEL.K(xa, self.XF)
        out = np.dot(ks, self.invCZ)

        E = np.random.normal(0, noise_std, xa.shape[0])

        return (out.reshape(-1, 1) + E.reshape(-1, 1))

    def generate_function(self):
        print("Generating test function")
        np.random.seed(self.seed)

        self.XF = np.random.uniform(size=(50, self.dxa)) * (self.xmax - self.xmin) + self.xmin


        mu = np.zeros(self.XF.shape[0])

        C = self.KERNEL.K(self.XF, self.XF)

        Z = np.random.multivariate_normal(mu, C).reshape(-1, 1)
        invC = np.linalg.inv(C + np.eye(C.shape[0]) * 1e-3)

        self.invCZ = np.dot(invC, Z)

    def check_input(self, x):
        if not x.shape[1] == self.dxa or (x > self.xmax).any() or (x < self.xmin).any():
            raise ValueError("x is wrong dim or out of bounds")
        return x

test_f = GP_test([0],[100])

# --- Attributes
#repeat same objective function to solve a 1 objective problem
f = MultiObjective([test_f,test_f])

# --- Space
#define space of variables
space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (0,3)}])

# --- Model (Multi-output GP)
#GP fit with access to gradients
n_a = 2
model = multi_outputGP(output_dim = n_a)

# --- Aquisition optimizer
#optimizer for inner acquisition function
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='sgd', space=space)
#
# # --- Initial design
#initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 15 )
print(initial_design)

#Training the GP model
X_inmodel = space.unzip_inputs(initial_design)
Y_new, _ = f.evaluate(initial_design)
Y_inmodel = list(Y_new)
model.updateModel(X_inmodel, Y_inmodel)

# --- Parameter distribution
#doesnt matter the value as long the sum of elements of support is equal to 1
support=[[0.5,0.5]]
prob_dist= [1]
parameter_distribution = ParameterDistribution(support=support, prob_dist=prob_dist)

#--- Utility function
def U_func(parameter,y):
    return np.dot(parameter,y)

def dU_func(parameter,y):
    return parameter

def log_file( **kwargs):
    '''
    dummy function. receives and outputs the same arguments. but allows writeCSV to pick up the names of the
    variables to be printed in the csv file
    '''
    return kwargs

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)
#
#
# --- Aquisition function
acquisition = maKG(model, space, optimizer=acq_opt , utility=U)

#Finally! computes acquisition function for each one of the points that you feed.
x_space = np.linspace(0,100,100)[:,None]
kg = acquisition._compute_acq(x_space)

#plots
plt.plot(x_space,kg)
plt.show()




