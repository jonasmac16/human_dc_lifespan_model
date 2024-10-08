projectdir_name = dirname(dirname(@__DIR__))

using Pkg
Pkg.activate(projectdir_name)

using DrWatson
DrWatson.@quickactivate "Model of DC Differentiation"

analysis_notebooks = readdir(projectdir("notebooks", "03_analysis", "02_uniform_prior"))

include(projectdir("scripts", "run_project", "00_threads.jl"))

for j in analysis_notebooks
    println("Started running " * j)
    println("...")
    file_tmp = projectdir("notebooks", "03_analysis", "02_uniform_prior", j, j*".jl")
    cmd_tmp = `julia +1.6.1 -t $n_threads $file_tmp`
    run(cmd_tmp)
    println("Finished running " * j)
    println("")
end