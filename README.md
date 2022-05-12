# Model of DC Differentiation

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Model of DC Differentiation

It is authored by Jonas Mackerodt as part of the following publiction:

> Amit A Patel, Ruth Lubi, Yan Zhang, Charles A Dutertre, Kevin Mulder, Jonas Mackerodt, Parinaaz Jalali, et al. ‘The Lifespan and Role of Human Dendritic Cell Subsets and Their Precursors in Health and Inflammation’

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently. Please contact the corresponding author with regards to this and copy the data into the `path/to/this/project/data/exp_raw` folder.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```
   This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box.
2. To further setup the enviroment follow either of the following two options: 
3. (Option 1) To finish setting up the enviroment, preprocess the data, run the inference and analyse the results run the following in the commandline (once navigated to `path/to/this/project/scripts/run_project`):
   ```
   julia 00_run.jl
   ```
4. (Option 2) To achieve the above in separate you can also run each julia script separately:
   ```
   julia 01_setup.jl
   julia 02_run_preprocessing.jl
   julia 03_run_fitting.jl
   julia 04_run_analysis.jl
   ```
5. To change the number of assigned threads to Julia adjust `00_threads.jl` accordingly and to change the number of parallel MCMC chains and number of samples obtained please adjust `00_mcmc_setting.jl` in in `scripts/run_project`.
