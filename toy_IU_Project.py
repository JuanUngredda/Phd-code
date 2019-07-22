import GPy
import csv
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
import time
import pygmo as pg
from scipy.stats import uniform 
from pyDOE import *
from scipy import optimize
import pandas as pd

import scipy.integrate as integrate
import scipy.special as special

import time
from scipy.stats import truncnorm

"""
toy code for Input Uncertainty results. This code represented the basis for the result of the IMA conference 
presentation 2019. Simulation optimisation, i.e., the search for a design or solution that optimises some output value 
of the simulation model, allows to automate the design of complex systems and has many real-world applications. 
Yet, several difficulties arise when dealing with real systems, specially long simulation running times, and stochastic 
outputs. Also, stochastic simulations take in probabilistic assumptions, through system logic, to produce random outputs
that must be estimated. Therefore, when constructing the simulation model, the decision maker often faces the
challenge of defining input distributions (eg. the mean of an arrival time distribution), in particular, if multiple
candidate distributions can fit the input data reasonably well, performance analysis are subjected to input error,
variability or uncertainty P[A|Data].

Moreover, if both, running additional simulations to learn about the output landscape mu(X,A), and collecting more 
data to reduce the input uncertainty P[A|Data] are expensive, then it is important to evaluate the trade-off 
between them since devoting too much effort to data collection (left image) may not leave sufficient time for 
optimisation, while devoting too little effort to data collection will require us to search for a robust solution 
that performs well across the possible input distribution, but may not be best for the true input parameters.
"""


# =============================================================
# Auxiliary functions for Gaussian Process posterior prediction and posterior Covariance
def MU(model, xa):
    K = model.kern.K(model.X, model.X)
    L = np.linalg.cholesky(K + (0.1 ** 2.0) * np.eye(len(K)))
    Lk = np.linalg.solve(L, model.kern.K(model.X, xa))
    mu = np.dot(Lk.T, np.linalg.solve(L, model.Y))
    return mu


def COV(model, xa1, xa2):
    K = model.kern.K(model.X, model.X)
    L = np.linalg.cholesky(K + (0.1 ** 2.0) * np.eye(len(K)))
    Lk1 = np.linalg.solve(L, model.kern.K(model.X, xa1))
    Lk2 = np.linalg.solve(L, model.kern.K(model.X, xa2))
    K_ = model.kern.K(xa1, xa2)
    s2 = np.matrix(K_) - np.matrix(np.dot(Lk2.T, Lk1))
    return s2


def COV1(model, xa1):
    K = model.kern.K(model.X, model.X)
    L = np.linalg.cholesky(K + (0.1 ** 2.0) * np.eye(len(K)))
    Lk = np.linalg.solve(L, model.kern.K(model.X, xa1))
    K_ = model.kern.K(xa1, xa1)
    s2 = K_ - np.sum(Lk ** 2, axis=0)
    return s2


def true_Q(a):
    X = lhs(1, samples=1000) * 100
    xa = [[i, a] for val in X for i in val]
    F_a = np.sum(test_func(xa, NoiseSD=0))
    return F_a


def predic_Q(a, model, Nx):
    X = lhs(1, samples=1000) * 100
    xa = [[i, a] for val in X for i in val]
    F_x = np.mean(model.predict(np.array(xa))[0])
    return F_x


def SUM_MU(X, a, model, Nx):
    xa = [[i, a] for val in X for i in val]
    F_a = np.sum(model.predict(np.array(xa))[0])
    return F_a


def SUM_COV(a, xan, model, Nx):
    X = lhs(1, samples=Nx) * 100
    xa = [[i, a] for val in X for i in val]
    COV = [model.kern.K(np.array([i]), np.array([xan])) for i in xa]
    SUM_COV = np.sum(COV)
    return SUM_COV


# ===============================================================

def test_func(xa, NoiseSD=0.1, seed=11,gen=False):
    """

    A toy function GP
    ARGS
     seed: int, RNG seed
     xa: n*d matrix, points in space to eval testfun
     NoiseSD: additive gaussaint noise SD
     gen: if set True, generates a gaussian process. If false, only evaluates at a specific design xa
    RETURNS
     output: vector of length nrow(xa)
    """
    np.random.seed(np.int(time.clock()*1000))
    KERNEL = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=([10,10]), ARD = True)
    if gen == True:
        np.random.seed(seed)
        Xt0=np.linspace(0,100,16) ; nX = len(Xt0)
        At0=np.linspace(0,100,15);  nA= len(At0)

        test_func.XtA0 = np.array([[i,j] for j in At0 for i in Xt0]) #FITS WITH MICHAEL
        mu = np.zeros(len(test_func.XtA0)) # vector of the means
        C = KERNEL.K(np.array(test_func.XtA0),np.array(test_func.XtA0)) # FITS WITH MICHAEL
        
        Z = np.random.multivariate_normal(np.zeros(len(C)), C) #np.array(pd.read_csv('/home/juan/Downloads/Individual_Project/Notebooks/Z.csv',index_col=False)['x'])
        #Z = np.array(pd.read_csv('/home/juan/Downloads/Individual_Project/Notebooks/Z.csv',index_col=False)['x'])
        invC = np.linalg.inv(C+np.eye(len(C))*1e-3)
        
        test_func.invCZ = np.dot(invC,np.matrix(Z).reshape(len(Z),1)) 
    
    
    ks = KERNEL.K(np.array(xa),np.array(test_func.XtA0))
    out = np.dot(ks,test_func.invCZ)
 
    E = np.random.normal(0,NoiseSD,len(xa))

    return (out.reshape(len(out),1) + E.reshape(len(E),1))

def Gen_func_2(Y =[],Pace=1):
    """
    Given a data set, uniform prior, and normal likelihood. Generates the marginal posterior probability
    of the parameter a. Therefore models P[A|Data]
    :param Y: narray of data
    :param Pace:
    :return: Dist: probability distribution pdf, Gen_Sample: generated samples of posterior pdf, Y_Sample: Predictive
    density samples P[ynew| data] given current data.
    """
    Gen_func_2.Y = Y
    precision = 1000
    def Distr_Update(M):


        def Data_Input_gen(N=1):
            """
            Generates data according to a data source
            :param N: Number of samples to generate
            :return: value of samples
            """
            if N <1:
                return []
            else:
                np.random.seed(np.int(time.clock()*1000))
                return np.random.normal(True_param,5,N)
        Yi = Data_Input_gen(M)
        Y = np.c_[[Gen_func_2.Y],[Yi]][0]

        #generates lattice
        MU = np.linspace(0,100,101)
        SIG = np.linspace(0.025,100,101)
        X = np.repeat(list(MU),len(SIG))
        W=list(SIG)*len(MU)
        MUSIG0 = np.c_[X,W]
        L = (np.exp(-(1.0/(2.0*MUSIG0[:,1]))*np.sum(np.array((np.matrix(MUSIG0[:,0]).T  - Y))**2.0,axis=1))*(1.0/np.sqrt(2*np.pi*MUSIG0[:,1]))**len(Y))
        L = np.array(L).reshape(len(MU),len(SIG))
        dmu = MU[1]-MU[0]
        dsig = SIG[1]-SIG[0]
        LN = np.sum(L*dmu*dsig)
        P = L/LN
        marg_mu = np.sum(P,axis=1)*dsig
        return marg_mu  ,Y
        
    def Gen_Sample(Dist,N=1):
        """
        Given a pmf generates samples assuming pmf is over equally
    spaced points in 0,...,100
        :param Dist: vector of probalilities
        :param N: sample size
        :return:
        """
        elements = np.linspace(0,100,len(Dist))
        probabilities = Dist/np.sum(Dist)
        val = np.random.choice(elements, N, p=probabilities)
        return val
        
    Dist ,Y_Sample= Distr_Update(Pace)
    Gen_Sample =Gen_Sample(Dist,N=100)
   
    Gen_Sample.sort()
    return Dist, Gen_Sample, Y_Sample

def DIF_Loss(model,y_sample,Dist):
    """
    Computes the benefit of sampling a new data point. Later compared with Knowleadge Gradient
    acquisition function
    :param model: surrogate model trained with data
    :param y_sample: current data narray
    :param Dist: current estimation for input uncertainty of the model narray.
    :return: scalar. Best expected improvement when a new data point is sampled.
    """
    X0 = np.linspace(0,100,100)
    precision = 1000
    
    def Gen_Sample(Dist,Y_SAMPLE,N=1,from_dist=True):
        if from_dist == True:
            elements = np.linspace(0,100,len(Dist))
            probabilities = Dist/np.sum(Dist)
            val = np.random.choice(elements, N, p=probabilities)
        else:
            
            mu, sigma = np.mean(Y_SAMPLE), np.sqrt(Sig2)/np.sqrt(len(Y_SAMPLE))
            a, b = (0 - mu) / sigma, (100 - mu) / sigma
            val= truncnorm.rvs(a, b, scale=sigma, loc=mu, size=N)
        val.sort()
        return val
    
    def Loss1(Sample):
   
        W0 =Sample
        X = np.repeat(list(X0),len(W0))
        W=list(W0)*len(X0)
        XW0 = np.c_[X,W]
        Prd = Project.m.predict(np.array(XW0))[0].reshape(len(X0),len(W0))

        IU = np.mean(Prd,axis=1)
      
        topX = X0[np.argmax(IU)]
        
        obj     = lambda a: np.mean(Project.m.predict(np.array(np.c_[np.repeat(a,len(W0)),W0]))[0])
        obj_min = lambda b: -obj(b)
        
        if topX >= 100:
            topX=99
        elif topX <=0:
            topX=1
            
        topX = np.longdouble(optimize.fminbound(obj_min, topX-1, topX+1,xtol =1e-8))

        IU_metric = Project.m.predict(np.c_[np.repeat(topX,len(W0)),W0])[0]
        REVI_= np.max(Prd,axis=0)
        #print('IU_metric',IU_metric,'REVI_',REVI_,'Dif',REVI_-IU_metric.T)
        DIFF1 = np.mean(REVI_-IU_metric.T)
    
        return DIFF1

    def Loss2_2():
        
        MU = np.linspace(0,100,101)
        SIG = np.linspace(0.025,100,101)
        X = np.repeat(list(MU),len(SIG))
        W=list(SIG)*len(MU)
        MUSIG0 = np.c_[X,W]
        Y =y_sample
        L = []
        fy = []
        y_n1 = np.linspace(0,100,101)
        for i in MUSIG0:
            fy.append(np.exp(-(1.0/(2.0*i[1]))*(i[0] - y_n1)**2.0))
            L.append(np.exp(-(1.0/(2.0*i[1]))*np.sum((i[0] - Y)**2.0))*(1.0/np.sqrt(2*np.pi*i[1]))**len(Y))
        dmu = MU[1]-MU[0]
        dsig = SIG[1]-SIG[0]
        dy_n1 = y_n1[1]-y_n1[0]

        L = np.matrix(L)
        fy = np.matrix(fy)
        D = np.array((np.matrix(L))*np.matrix(fy)*dmu*dsig)
        D = np.array((D/np.sum(D*dy_n1)))[0]
        
        Dw = Gen_Sample(D,y_sample,N=100,from_dist=True)

        y_samp = np.ones(len(Dw)).reshape(len(Dw),1)*y_sample
        U = np.c_[y_samp,Dw]  
       
        R_IU = []
        
        for i in U:
            
            #W0 = Gen_Sample(Dist,i,N=100,from_dist=False)
            W0 = Gen_func_2(i,Pace=0)[1]
            
            X = np.repeat(list(X0),len(W0))
            W=list(W0)*len(X0)
            XW0 = np.c_[X,W]
            Prdtn = Project.m.predict(np.array(XW0))[0].reshape(len(X0),len(W0))
            IU = np.mean(Prdtn,axis=1)

            topX = X0[np.argmax(IU)]
            
            obj     = lambda a: np.mean(Project.m.predict(np.array(np.c_[np.repeat(a,len(W0)),W0]))[0])
            obj_min = lambda b: -obj(b)
            
            topX = np.longdouble(optimize.fminbound(obj_min, topX-1, topX+1,xtol =1e-8))
    
            IU_metric = Project.m.predict(np.array(np.c_[np.repeat(topX,len(W0)),W0]))[0]

            REVI_metric =np.max(Prdtn,axis=0)
            REVI_IU_metric = (REVI_metric - IU_metric.T)
            
            R_IU.append(REVI_IU_metric)
        DIF_Loss.R_IU = R_IU

        return np.mean(R_IU)
    

    if len(y_sample) == 0:
        Sample = np.random.random(1000)*100
    else:
        Sample = Gen_Sample(Dist,y_sample,N=1000,from_dist=True)
    
#     plt.plot(np.linspace(0,100,len(Dist)),Dist)
#     plt.hist(Sample,bins=100,range=(0,100),normed=True)
#     plt.show()
    
    L1 =Loss1(np.array(Sample))
    #L2 = Loss2()
    L2_2 = Loss2_2()
    #DIFF =   L1 - L2
    DIFF2 = L1 - L2_2
    print('Loss1',L1,'Loss2_2',L2_2,'Diff2',DIFF2)
    #print('Loss1',L1,'Loss2',L2,'Diff',DIFF)
    return DIFF2



def Project(Start_S=10,Stop_S=100,True_param = 50):
    """
    Main function for project. Here is implemented the main loop where knowleadge gradient is compared with
    Delta Loss to determine if either do a an optimisation run or get a new data source
    :param Start_S: number of point of initial design to train gaussian process
    :param Stop_S: number of iterations before stopping the algorithm
    :param True_param: True parameter value from data source
    :return: OC: Difference between the best true value and the value recommended by the expected performance
    """
    def UNIdesign(Nxa=1):
        """
        Generates uniform design over lattice XA and samples from the test function
        :param Nxa: number of points to generate
        :return: value of generated points
        """
        np.random.seed(np.int(time.clock()*10000))
        H_true = [10,10,1]
        X = lhs(1, samples=Nxa)*100
        A = lhs(1, samples=Nxa)*100
        XA = np.array(np.c_[X,A])
        Y = test_func(XA)
        return XA,Y
    
    def method(XA,P,model,method =1, Nx=15,Ns=20):
        if method ==2:
            print('KG_MC_Input...')
            return KG_Mc_Input(XA,P,model, Nx,Ns)        
        if method ==3:
            print('RANDOM SAMPLING..')
            return UNIdesign(Nxa=1)      


    def KG_Mc_Input(XA,P,model, Ns=20):
        """
        This is Knowleadge gradient for correlated beliefs implemented as Powel & Frazier.
        This acqusition function is compared with Delta Loss
        :param XA: lattice between one design and one inpu
        :param P:
        :param model: GP model trained with data
        :param Ns: Number of points for internal optimisation process of KG
        :return: scalar. represents the expected improvement taking future corraltions changes.
        """
        global best_obj_v
        np.random.seed(np.int(time.clock()*10000))
        KG_Mc_Input.Ns = Ns
        KG_Mc_Input.bestEVI = -10
        KG_Mc_Input.bestxa  = [-10,-10]
        KG_Mc_Input.noiseVar = model.Gaussian_noise.variance[0] 
        KG_Mc_Input.Xi = np.array(XA[:,0])
        KG_Mc_Input.Xi.sort()

        KG_Mc_Input.Ad = Tsample #1000

        X = np.repeat(list(KG_Mc_Input.Xi),len(KG_Mc_Input.Ad ))
        A=list(KG_Mc_Input.Ad )*len(KG_Mc_Input.Xi)
        KG_Mc_Input.XiAd  = np.c_[X,A]
   
        
        obj_v = np.sum(np.array(m.predict(KG_Mc_Input.XiAd)[0]).reshape(len(KG_Mc_Input.Xi),len(KG_Mc_Input.Ad)),axis=1)
        obj_v = np.array(obj_v).reshape(len(obj_v),1)

        def KG_IU(xa):
       
            if(np.abs(xa[0]-50)>50 or np.abs(xa[1]-50)>50 ):
                return(1000000)
            else:

                dXi     =  1
           
                A_a = np.concatenate(([xa[1]],KG_Mc_Input.Ad))
                X_x = np.concatenate(([xa[0]],KG_Mc_Input.Xi))
                newx = np.array(np.c_[np.repeat(xa[0],len(A_a)),A_a])
                newa = np.array(np.c_[X_x,np.repeat(xa[1],len(X_x))])
                
                MMa = np.sum(model.predict(newa)[0])
                
                MM     =  (np.c_[MMa,obj_v.T].reshape(len(np.c_[MMa,obj_v.T].T),1) + dXi*model.predict(newa)[0])
                MM = MM/(len(MM)+1)
         
                sigt2 = np.diag(COV(m,np.array([xa]),KG_Mc_Input.XiAd)).reshape(len(KG_Mc_Input.Xi),len(KG_Mc_Input.Ad))
        
                sigt2 = np.array(np.sum(sigt2,axis=1)).reshape(len(sigt2),1)
                sigt2 = np.c_[np.sum(np.diag(COV(m,np.array([xa]),newa))),sigt2.T].T

                sigt3 = np.array(np.diag(COV(m,np.array([xa]),newa)))
                sigt3 = sigt3.reshape(len(sigt3),1)

                sigt1   = COV1(model,np.array([xa]))
                sigt    = ((sigt2 + sigt3*dXi ) / np.sqrt(sigt1+KG_Mc_Input.noiseVar))
                sigt = sigt/(len(sigt)+1)
                musig = np.c_[MM,sigt]

                out  = KG(musig)
                if out > KG_Mc_Input.bestEVI:
                    KG_Mc_Input.bestEVI = out
                    KG_Mc_Input.bestxa = xa
                return -out
#             #          ##GRAPHHHHHH

        XAs = np.array(np.c_[lhs(1, samples=KG_Mc_Input.Ns)*100,lhs(1, samples=KG_Mc_Input.Ns)*100])
        start1 = time.time()
    
        A    = [minimize(KG_IU, i, method='nelder-mead', tol= 1e-8).x for i in XAs]
        Y = test_func(np.array([list(KG_Mc_Input.bestxa)]))
        done1 = time.time()
        print('bestxa',np.array([list(KG_Mc_Input.bestxa)]),'Y',np.array(Y))
        print('Opt_time',done1-start1)
        return np.array([list(KG_Mc_Input.bestxa)]) , np.array(Y)


    #=============================================================================================
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #==============================================================================================

    X0=np.linspace(0,100,30)
    A0=np.linspace(0,100,30)
    X = np.repeat(list(X0),len(A0))
    A=list(A0)*len(X0)
    XA0 = np.c_[X,A]
    P = test_func(xa=XA0,gen=True,seed=20)
    StartN = Start_S
    EndN = Stop_S

    ker = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=([10,10]), ARD = True)
    XA, P = UNIdesign(StartN)
    iLoss = []
    OC = []
    #===================================================================================
    #True Func
    Input_Dist ,Tsample,y= Gen_func_2(Pace=0)
    print('Initial Input Distribution')
    print('y sample',y,'Sample size',len(Tsample))

    X0= np.linspace(0,100,100)
    W0= [True_param]#np.random.normal(50,15,1000)
    X = np.repeat(list(X0),len(W0))
    W=W0*len(X0)
    XW0 = np.c_[X,W]

    Pr = test_func(xa=XW0,NoiseSD=0)
    obj     = lambda a: np.mean(test_func(np.array(np.c_[np.repeat(a,len(W0)),W0]),NoiseSD=0))
    obj_min = lambda b: -obj(b)

    True_obj_v = np.mean(Pr.reshape(len(X0),len(W0)),axis=1)

    topX    = X0[np.argmax(True_obj_v)]
    topX = optimize.fminbound(obj_min, topX-1, topX+1,xtol =1e-16)

    best_ = obj(topX)
    print('max',best_,'argmax',topX)
    #=================================================================================================

    while len(XA) + len(y)< EndN:
        m = GPy.models.GPRegression(np.array(XA) , np.array(P).reshape(len(P),1) , ker,noise_var=0.1**2.0)
        Project.m = m
        #=================================================================================================
        #ERROR CURVE

        X0=np.linspace(0,100,100)
        A0=Tsample
        X = np.repeat(list(X0),len(A0))
        A=list(A0)*len(X0)
        XA0 = np.c_[X,A]

        obj     = lambda a: np.mean(m.predict(np.array(np.c_[np.repeat(a,len(A0)),A0]))[0])
        obj_min = lambda b: -obj(b)
        obj_v = np.mean(np.array(m.predict(np.array(XA0))[0]).reshape(len(X0),len(A0)),axis=1)
        topX    = X0[np.argmax(obj_v)]
        Xr = np.longdouble(optimize.fminbound(obj_min, topX-1, topX+1,xtol =1e-12))
        W_test = np.array([True_param])


        topobj = np.longdouble(np.mean(test_func(np.array(np.c_[np.repeat(Xr,len(W_test)),W_test]),NoiseSD=0)))
        DIF = np.longdouble(best_ - np.longdouble(topobj))
        print('best',best_,'max',topobj,'argmax',Xr,'DIF',DIF)
        OC.append(DIF)
        #===============================================================================================
        DF = DIF_Loss(Project.m,y,Input_Dist)
        xa, p = method(XA,P,model = m,method=2,Nx=len(XA),Ns=10)

        print('Difference Loss',DF,'Best KG',KG_Mc_Input.bestEVI)
        """
        Comparison between Konleadge gradient and Delta Loss
        """
        if KG_Mc_Input.bestEVI > DF:
            print('Improve Sample from surface')
            XA = np.concatenate((XA,xa)) ; P = np.concatenate([P,p])
        else:
            iLoss.append(DF)
            Input_Dist ,Tsample,y= Gen_func_2(y)

            print('Improved Input Distribution....Initial Input Distribution')
            print('y sample',y,'mean_sample',np.mean(Tsample))

        print('n(XA)',len(XA),'y sample len',len(y))
    return OC

if __name__ == "__main__":


OC= []

for i in range(7,100):
    True_param=np.random.random()*100
    print('True_param',True_param)
    OC = Project(Start_S=10,Stop_S=100,True_param=True_param)

    data = {'OC': OC}
    gen_file = pd.DataFrame.from_dict(data)
    path ='/home/rawsys/matjiu/PythonCodes/RESULTS/lechon_OC_'+str(i)+'.csv'
    gen_file.to_csv(path_or_buf=path)
