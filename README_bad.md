##toy_IU_project.py

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

Slides of IMA presentation can be found in https://warwick.ac.uk/fac/sci/mathsys/people/students/2017intake/ungredda/

##full_bayes_EI.py

Developed code to perform Bayesian optimisation using Expected Improvement as an acquisition function and Probability
of Feasibility to adapt the constrains of the problem. This code marginalise the hyperparameters after the mcmc
sampling using Slice Sampling yields several samples from the posterior distribution. 