import numpy as np
import GPy
import matplotlib.pyplot as plt
from optimizer.discrete_KG import KG_optimizer

#generates random gp
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

#init data X as inputs Y as outputs
X = np.random.random((10,1))*100
Y = test_f(X)

#init GP model with right hps
ker = GPy.kern.RBF(input_dim=1, variance=4, lengthscale=10, ARD=True)
model = GPy.models.GPRegression(X, Y, ker, noise_var=0.01)

#input space to calculate KG at each point
x_space = np.linspace(0,100,1)[:,None]

#Call KG acquisition function
optimizer = KG_optimizer(model)
kg = optimizer(x_space)
print("kg",kg)
#Plots
plt.scatter(X,Y)
plt.plot(x_space,np.array(kg))
plt.show()





