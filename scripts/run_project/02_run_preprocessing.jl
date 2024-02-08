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
    cmd_tmp = `julia +1.6.1 -t $n_threads $file_tmp`
    run(cmd_tmp)
    println("Finished running " * j)
    println("")
end