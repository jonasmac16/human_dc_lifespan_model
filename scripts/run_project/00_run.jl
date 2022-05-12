projectdir_name = dirname(dirname(@__DIR__))

include(joinpath(projectdir_name, "scripts", "run_project", "01_setup.jl"))
include(joinpath(projectdir_name, "scripts", "run_project", "02_run_preprocessing.jl"))
include(joinpath(projectdir_name, "scripts", "run_project", "03_run_fitting.jl"))
include(joinpath(projectdir_name, "scripts", "run_project", "04_run_analysis.jl"))