projectdir_name = dirname(dirname(@__DIR__))

using Pkg
Pkg.activate(projectdir_name)

using DrWatson
DrWatson.@quickactivate "Model of DC Differentiation"

processing_notebooks = readdir(projectdir("notebooks", "01_processing"))

include(projectdir("scripts", "run_project", "00_threads.jl"))

for j in processing_notebooks
    println("Started running " * j)
    println("...")
    file_tmp = projectdir("notebooks", "01_processing", j, j*".jl")
    cmd_tmp = `julia -t $n_threads $file_tmp`
    run(cmd_tmp)
    println("Finished running " * j)
    println("")
end

# include(projectdir("notebooks", "01_processing", "JM_0002_Julia_Fitting_bodywater_enrichment", "JM_0001_Julia_Import_and_processing_datasets-pluto.jl"))
# include(projectdir("notebooks", "01_processing", "JM_0003_Julia_Fitting_Proliferation_priors", "JM_0001_Julia_Import_and_processing_datasets.jl"))