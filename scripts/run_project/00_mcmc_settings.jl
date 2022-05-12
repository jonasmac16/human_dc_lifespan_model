mcmc_iters = 300
n_chains = 4
parallel_sampling_method = MCMCThreads() #nothing or MCMCThreads()
solver_parallel_methods = EnsembleThreads() #EnsembleSerial() or #EnsembleThreads() for parallel solving of ODEs