projectdir_name = dirname(dirname(@__DIR__))

using Pkg
Pkg.activate(projectdir_name)
Pkg.instantiate()


ENV["R_HOME"] = "*"
Pkg.build("RCall")

using Conda
Conda.add("r-loo", channel = "conda-forge")