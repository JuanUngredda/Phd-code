import numpy as np
#from matplotlib import pyplot as plt
import GPy
from scipy.stats import multivariate_normal
from scipy import optimize
import scipy

#==================================================================
#Auxiliary Functions of the main code
"""
Slice sampler implemented in python from Murray & Adams "Slice sampling covariance hyperparameters
of latent Gaussian models". 
"""

def evaluate_particle(DATA, theta_vector, n,gg_vector,alpha,ff=0,verbose=False): 
    
    Ltprior = log_priors(theta_vector)

    if Ltprior != -np.inf:
        
        #L = scipy.linalg.cho_factor(Kernel)
        k,S_theta, Kernel, invchol_Kernel = update_kernel(DATA,theta_vector,alpha)
        
        X = DATA[:,0][:,None]
        Y = np.vstack(DATA[:,1])
        
        Sinvg = np.vstack(gg_vector*(1.0/alpha))
        ff_vector = (np.linalg.solve(invchol_Kernel.T,n) + scipy.linalg.cho_solve((invchol_Kernel.T,False),Sinvg))

        L_Kernel = np.linalg.cholesky(Kernel)
        Lfn = log_likelihood(Y,ff_vector ,alpha)

        Lfprior = -0.5*np.dot(ff_vector.T,scipy.linalg.solve(Kernel,ff_vector))- np.sum(np.log(np.diag(L_Kernel))) -(len(Y)/2.0)*np.log(2*np.pi)

        LJacobian = -np.sum(np.log(np.diag(invchol_Kernel)))

        Lg_f = log_likelihood(gg_vector,ff_vector ,alpha)
        Lp_total = Lfn + Lfprior + LJacobian + Lg_f + Ltprior

#         if verbose == True:
#             print('evaluated theta vector',np.exp(theta_vector))
#             plt.scatter(X,ff_vector,label='evaluated_thetas')
#             plt.scatter(X,DATA[:,1],label='data')
#             plt.scatter(X,gg_vector,label='gg')
#             plt.legend()
#             plt.show()
#             #print(multivariate_normal.logpdf(x=ff_vector.T[0],mean=np.zeros(len(ff_vector)),cov=Kernel))
#             print('Lfn, Lfprior, LJacobian, Lg_f, Ltprior',Lfn, Lfprior, LJacobian, Lg_f, Ltprior)
#             print('total',Lp_total)
            
        evaluate_particle.ff_vector = ff_vector
        return Lp_total
    else:
        #evaluate_particle.ff_vector = ff_vector
        Lp_total = -np.inf
        return Lp_total

def log_prior(DATA , Kern):
    Y = np.vstack(DATA[:,1])
    lp = -0.5*np.dot(Y.T,np.linalg.solve(Kern,Y))-(len(Y)/2.0)*np.log(2*np.pi) - np.log(np.sum(np.diag(np.linalg.cholesky(Kern))))
    return lp

def log_priors(theta_vector):
    max_ls         = 10.0;
    min_ls         = 0.01;
#     print('theta_vector',theta_vector)
    Ltprior = np.log(1.0*all((theta_vector>np.log(min_ls)) & (theta_vector< np.log(max_ls))))
    return Ltprior

def log_likelihood(DATA,f_sample,sigma): 
    Y = np.vstack(DATA)
    f = np.vstack(f_sample)
    llh = -0.5*np.sum((1.0/sigma)*(f - Y)**2.0)-0.5*np.log((sigma**len(Y))*(2*np.pi)**(len(Y)))
    return llh

def update_kernel(DATA,theta_vect,alpha,jitter=1e-5):
    k = GPy.kern.RBF(1)
    X = DATA[:,0][:,None] 
    
    k.lengthscale = np.exp(theta_vect[0]) 
    k.variance = np.exp(theta_vect[1])
    
    S_theta = np.identity(len(X))*alpha
    Kernel = k.K(X,X) + np.identity(len(X))*jitter
    
    #R = np.linalg.inv(np.linalg.inv(S_theta) + np.linalg.inv(Kernel))
    invchol_Kernel = np.linalg.cholesky(np.linalg.inv(S_theta) + np.linalg.inv(Kernel))
    

    return k,  S_theta, Kernel, invchol_Kernel

def SLICE_SAMPLER(DATA,burn_in, Ns, verbose=False):
    def ell_ss(tht,sf,DATA, alpha,verbose=False):
        #INIT VALUES INSIDE COV

        #choose elipse
        z = np.random.normal(0,1,len(C))[:,None]
        nu = np.dot(np.linalg.cholesky(C), z).T[0]

        #loglike threshold
        u = np.random.random()
        ly_right = log_likelihood(DATA[:,1],sf,alpha)+np.log(u) 

        #draw proposal 
        phi = np.random.random()*np.pi*2
        phi_min = phi - 2*np.pi
        phi_max = phi


        while True:
            
            esl_f = sf*np.cos(phi) + nu*np.sin(phi)
            
            ly_left = log_likelihood(DATA[:,1],esl_f,alpha)
            if ly_left > ly_right:
                return esl_f
            else:
                if phi > 0:
                    phi_max = phi
                else:
                    phi_min = phi
            phi = np.random.random()*(phi_max-phi_min) + phi_min
        

    def seed():
        k = GPy.kern.RBF(1)
        m = GPy.models.GPRegression(DATA[:,0][:,None],DATA[:,1][:,None])
        m.optimize_restarts(1, robust=True, verbose=True)
        
#         if m.rbf.lengthscale > 30 or m.rbf.variance > 30:
#             sample_l = np.log(m.rbf.lengthscale)
#             sample_v = np.log(m.rbf.variance)
#         else:
        sample_l = 0
        sample_v = 0
        
        if  m.Gaussian_noise.variance < 1e-9:
            m.Gaussian_noise.variance = 1e-4
        alpha = m.Gaussian_noise.variance
        k.lengthscale=np.exp(sample_l)
        k.variance = np.exp(sample_v)
        
        C  = k.K(X,X) + np.identity(len(X))*jitter
        f  = np.random.multivariate_normal(np.zeros(len(X)), C , 1)[0]
        return [sample_l,sample_v] , f,alpha


    #INIT VALUES
    scale = 10
    Nhyp = 2
    
    #INIT DATA AND ARRAYS
    X = DATA[:,0][:,None]
    X_plot = DATA[:,0]
    grid = len(X)
    sample_theta = np.zeros((burn_in+Ns+1,Nhyp))
    sample_f = np.zeros((burn_in+Ns+1,grid))
    
    
    ############ INIT COV ###############
    
    jitter = 1e-5
    sample_theta[0],sample_f[0], alpha = seed()
    

    for ii in range(burn_in+Ns):
        #print('it........................................................',ii)
        stg1_tht,stg1_ff = update_thetas(DATA = DATA, ff=sample_f[ii], thts = sample_theta[ii], jitter = jitter,alpha=alpha ,verbose=True)
        k ,  S_theta , C , invchol_Kernel  = update_kernel(DATA,stg1_tht,alpha)
        for jj in range(10):
             stg1_ff = ell_ss(tht = stg1_tht , sf = stg1_ff , DATA=DATA, alpha=alpha ,verbose=False)
        sample_theta[ii+1],sample_f[ii+1] = stg1_tht, stg1_ff
    return sample_theta[burn_in:burn_in+Ns],sample_f[burn_in:burn_in+Ns]

def update_thetas(DATA, ff, thts, jitter = 1e-5,alpha=0,verbose=True):

    def sweep_slice(tht_sw_vect,min_threshold,step_out = True):

        Nhyp = 2
        scale = 5
        w = np.repeat(scale,Nhyp)
        aux_tht = tht_sw_vect
        for dd in np.random.permutation(Nhyp):
            x_cur = aux_tht[dd]
            rr = np.random.random()
            x_l = x_cur - rr*w[dd]

            x_r = x_cur + (1-rr)*w[dd]
            if step_out:

                aux_tht[dd] = x_l
                bug_counter = 0
                while True:
                    bug_counter+=1
                    if bug_counter > 500:
                        print('WARNING: BUG FOUND')
                        print('alpha',alpha)
                        print('x_l ',x_l,' x_cur ',x_cur,' x_r ',x_r)
                        print('aux_tht',aux_tht)
                        print('min_threshold',min_threshold)
                        print('particle_val',particle_val)
                    particle_val = evaluate_particle(DATA,aux_tht,n,g,alpha)
                    if min_threshold > particle_val:

                        break;
                    aux_tht[dd] = aux_tht[dd] - w[dd]

                x_l = aux_tht[dd]

                aux_tht[dd] = x_r
                bug_counter = 0
                while True:
                    bug_counter+=1
                    if bug_counter > 500:
                        print('WARNING: BUG FOUND')
                        print('alpha',alpha)
                        print('x_l ',x_l,' x_cur ',x_cur,' x_r ',x_r)
                        print('aux_tht',aux_tht)
                        
                    particle_val = evaluate_particle(DATA,aux_tht,n,g,alpha)

                    if min_threshold > particle_val:
                        break;
                    aux_tht[dd] = aux_tht[dd] + w[dd]

                x_r = aux_tht[dd]
                bug_counter = 0
                while True:
                    bug_counter+=1
                    if bug_counter > 500:
                        print('WARNING: BUG FOUND')
                        print('alpha',alpha)
                        print('x_l ',x_l,' x_cur ',x_cur,' x_r ',x_r)
                        print('aux_tht',aux_tht)
                        
                    aux_tht[dd] = np.random.random()*(x_r - x_l) + x_l
                    particle_val = evaluate_particle(DATA,aux_tht,n, g,alpha)

                    if particle_val> min_threshold:
                        break  #% Only way to leave the while loop.
                    else:
                        #% Shrink in
                        if aux_tht[dd] > x_cur:
                            x_r = aux_tht[dd];
                        elif aux_tht[dd] < x_cur:
                            x_l = aux_tht[dd];
                        else:
                            print('BUG DETECTED: Shrunk to current position and still not acceptable.');
        
        return aux_tht
    
#     print('INPUT thts',thts)
    k , S_theta, C,invchol_R = update_kernel(DATA, thts, alpha)#, jitter=jitter)        


    # DRAW SURROGATE DATA g|f,S
    X = DATA[:,0][:,None] #remove later
   
    
    ff = np.vstack(ff)
    g = ff + np.vstack(np.random.multivariate_normal(np.zeros(len(ff)),np.identity(len(ff))*alpha))


    Sinvg = np.vstack(g*(1.0/alpha))

    # WHITENING PRIOR

    #chol_R = np.linalg.cholesky(R)
    
    n = np.dot(invchol_R.T,ff)-np.linalg.solve(invchol_R,Sinvg)

#     #THRESHOLD

    min_threshold = evaluate_particle(DATA,thts, n,g,alpha,ff=ff,verbose=True)
    min_threshold = np.log(np.random.random())+min_threshold
    update_thetas.min_threshold = min_threshold

    #PROPOSE THETA NEW PARAMETERS

    #DECISION

    new_thts = sweep_slice(thts,min_threshold,step_out = True)

    return new_thts , evaluate_particle.ff_vector.T[0]

#============================================================================

def Bayes_Optimizer(init_sample,budget,method,test_f_num=1,SEEDS=1):
    """
    Main function. This function calculates Expected improvement and then marginilise the hyperparameters of
    the gaussian process to perform a fully bayesian inference
    :param init_sample: initial number of samples to train the gaussian process
    :param budget: number of evaluations allowed in the test function
    :param method: "BAYES" performs a fully Bayesian marginalisation of hyperparameters. Else, would perform
    classic Maximum likelihood estimates with several restarts
    :param test_f_num: gp test function that will be generated
    :param SEEDS: random seed for the inner optimisation
    :return: OC. Oportunity cost. Diffference between true best design and the lne recommended by the optimisation
    process. narray of OC across iterations.
    """

    def Optimisation(gp,thts):
        def Objective_Function(X ,Y_sample, gpr,xi=0):
            ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model.
            Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d).
            Y_sample: Sample values (n x 1).
            gpr: A GaussianProcessRegressor fitted to samples.
            xi: Exploitation-exploration trade-off parameter.
            Returns: Expected improvements at points X.
            '''
            #print('X',X)
            mu, sigma = gpr.predict(np.vstack(X), include_likelihood=False)
            mu_sample_opt = np.min(Y_sample)
            imp = mu_sample_opt - mu
            Z = imp / sigma
            ei = imp * scipy.stats.norm.cdf(Z) + sigma * scipy.stats.norm.pdf(Z)
            return -1*ei

        def acc(x):
            if x>f.u_limit or x<f.l_limit:
                return 10000
            else:
#                 print('x',x,'E_EI(x)',E_EI(x))
#                 print('Optimisation.x_val',Optimisation.x_val)
                if E_EI(x)<Optimisation.x_val:
                    Optimisation.x_val = E_EI(x)
                    Optimisation.x_init = x
                return E_EI(x)

        x_plot = np.linspace(0,10,1000)[:,None]
        Ns = 5
        
        Y_sample = gp.Y

        Optimisation.x_init = -10
        Optimisation.x_val = -10000
        
        if method == 'BAYES':
            Optimisation.x_init = -10
            Optimisation.x_val = 10000
        
            XAs = np.random.random(Ns)*10
            EI_function = []
            for i in np.exp(thts):
                m.rbf.lengthscale = i[0]
                m.rbf.variance = i[1]
                m.Gaussian_noise.variance = alpha
                EI_function.append(Objective_Function(x_plot, Y_sample, m,xi=0))
            
            mean_EI_function = np.mean(EI_function,axis=0)

            E_EI = scipy.interpolate.interp1d(x_plot.T[0], mean_EI_function.T[0])
            
#             plt.plot(x_plot,E_EI(x_plot),label='acc')
#             plt.legend()
#             plt.show()
            
            A    = [scipy.optimize.minimize(acc, np.array([[i]]), method='nelder-mead', tol= 1e-8).x for i in XAs]
        else:
            mean_EI_function = np.vstack(Objective_Function(x_plot, Y_sample, m,xi=0))
            Optimisation.x_init = -50
            Optimisation.x_val = 10000
            XAs = np.random.random(Ns) + (x_plot[np.argmin(mean_EI_function)]-0.5)
            for i in XAs:
                x_opt   = scipy.optimize.minimize(Objective_Function, np.array([[i]]), args =(Y_sample, m) ,method='nelder-mead', tol= 1e-12).x
                if x_opt < f.l_limit or x_opt > f.u_limit:
                    EI_eval = 10000
                elif x_opt > f.l_limit or x_opt < f.u_limit:
                    EI_eval = Objective_Function(np.array([[i]]),Y_sample, m)

                if EI_eval < Optimisation.x_val:
                        Optimisation.x_val = EI_eval
                        Optimisation.x_init = x_opt

        return np.array([Optimisation.x_init])
    
    DATA = np.zeros((budget+N,2))
    
    f.init(test_f_num,SEEDS,init_sample)

    DATA[0:N,0] = f.generate_data()
    DATA[0:N,1] = f.evl(DATA[0:N,0])
    f.data(DATA[0:N,0],DATA[0:N,1])
    #f.verbose()
    #MAXIMUM LIKELIHOOD ESTIMATION
    m = GPy.models.GPRegression(DATA[0:N,0][:,None],DATA[0:N,1][:,None])
    
    if method == 'BAYES':

        sample_theta = SLICE_SAMPLER(DATA[0:N,:],burn_in=10, Ns=10, verbose=False)[0]
        m.optimize_restarts(5, robust=True, verbose=False)
        alpha = m.Gaussian_noise.variance
        

    else:
        m.optimize_restarts(5, robust=True, verbose=False)
        
    CC = np.zeros(budget)
    for bb in range(budget):
        m = GPy.models.GPRegression(DATA[0:N+bb,0][:,None],DATA[0:N+bb,1][:,None])
        if method == 'BAYES':
            dsgn = Optimisation(gp = m ,thts=sample_theta)
        else:
            dsgn = Optimisation(gp = m ,thts=0)
#         print('dgn',dsgn)
        DATA[N+bb,0] = dsgn
        DATA[N+bb,1] = f.evl(dsgn)
        f.data(np.array([DATA[N+bb,0]]),np.array([DATA[N+bb,1]]))
    return


class test_func:
    """
    GP test function
    arg:
        seed: random seed
        init_sample: initial sample size
    """
    def init(self,seed,gen_seed,init_sample):
        self.N = init_sample
        self.EVAL = 9000
        self.l_limit = 0
        self.u_limit = 10
        
        self.seed_f_test = seed
        self.seed_X_gen = gen_seed
        self.set_rdm_test = np.random.seed(self.seed_f_test)
        
        self.k = GPy.kern.RBF(1)
        self.k.lengthscale = 1
        self.k.variance = 1
        self.X = np.linspace(0,10,50)[:,None]
        self.DATA_X = np.array([])
        self.DATA_Y = np.array([])
        self.K = self.k.K(self.X,self.X)
        self.f = np.dot(self.K,np.vstack(np.random.multivariate_normal(np.zeros(len(self.K)),self.K)))
        
        
    def evl(self,x):
        self.x = np.vstack(x)
        np.random.seed(self.seed_f_test)
        self.K_eval = self.k.K(self.x,self.X) 
        f_eval = np.dot(self.K_eval,np.vstack(np.random.multivariate_normal(np.zeros(len(self.K)),self.K)))
        return f_eval.T[0]
    
    def minimise_test(self):
        XAs = np.random.random(5)*10
        for i in XAs:
            A = scipy.optimize.minimize(self.evl, i, method='nelder-mead', tol= 1e-12).fun
            if A < self.EVAL:
                self.min_val = A
        
    def get_min(self):
        return self.min_val
    
    def data(self,X,Y):
        X = np.array(X)
        Y = np.array(Y)
        self.DATA_X = np.concatenate((self.DATA_X, X))
        self.DATA_Y= np.concatenate((self.DATA_Y, Y))
        

        
    def generate_data(self):
        self.seed_X_gen = np.random.seed(self.seed_X_gen)
        return np.random.random(self.N)*10
    
    def CC(self):
        self.minimise_test()
        prfmnc = np.min(self.DATA_Y)-self.min_val
        return prfmnc


import pandas as pd


RUNS = 2 #Number of repetitions of the whole budget
budget = 16 #number of evaluations of the test function
N=4 #Number of initial designs

#initialising lists and arrays to store results
CC_full_bayes = np.zeros((RUNS,1))
DATA_full_bayes = np.zeros((RUNS,budget+N,2)) # Data acquired durin the run

SEEDS = np.random.random_integers(1,100,RUNS)

for rr in range(RUNS):
    print('rr.....',rr)
    f = test_func()
    Bayes_Optimizer(init_sample = N , budget=budget,method='BAYES',test_f_num=1,SEEDS=SEEDS[rr])
    CC_full_bayes[rr,:] = f.CC()

#save path of results
path ='/home/juan/Documents/PhD/Tutorials/Monte-Carlo_Methods/RESULTS/OC_FB.csv'
np.savetxt(path,  CC_full_bayes, delimiter=",")
