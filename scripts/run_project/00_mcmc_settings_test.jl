mcmc_iters = 400
n_chains = 2
parallel_sampling_method = MCMCThreads() #nothing or MCMCThreads()
solver_parallel_methods = EnsembleThreads() #EnsembleSerial() or #EnsembleThreads() for parallel solving of ODEs