projectdir_name = dirname(dirname(@__DIR__))

using Pkg
Pkg.activate(projectdir_name)

using DrWatson
DrWatson.@quickactivate "Model of DC Differentiation"

fitting_notebooks = readdir(projectdir("notebooks", "02_fitting", "02_uniform_prior"))

include(srcdir("parallel_run.jl"))
# include(scriptsdir("run_project", "00_mcmc_settings.jl"))
include(scriptsdir("run_project", "00_threads.jl"))

parallel_task = divrem(Sys.CPU_THREADS-3,n_threads)[1]

function build_cmd(notebook_dir; load_file = nothing)
    notebook_full_path = projectdir("notebooks", "02_fitting", notebook_dir, notebook_dir*".jl")

    return isnothing(load_file) ? `julia +1.6.1 -t $n_threads $notebook_full_path` : `julia +1.6.1 -t $n_threads --load $load_file $notebook_full_path`
end

commands_arr = [build_cmd(j) for j in fitting_notebooks]

parallel_run(commands_arr; ntasks = parallel_task)