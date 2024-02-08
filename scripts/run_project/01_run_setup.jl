projectdir_name = dirname(dirname(@__DIR__))

using Pkg
Pkg.activate(projectdir_name)
Pkg.instantiate()

using Conda
Conda.add("mamba")
ENV["CONDA_JL_CONDA_EXE"] = joinpath(Conda.ROOTENV, "bin", "mamba")
Pkg.build("Conda")

# build PyCall
ENV["PYTHON"] = ""
Pkg.build("PyCall")


# Install important libraries and (make sure conda is v 23.1.0)
Conda.add("r-base")
Conda.add("r-loo")
Conda.add("matplotlib")


# build RCall
ENV["R_HOME"] = "/.julia/conda/3/lib/R/"
# Potentially you need to link to theup-to-date system libstdc++.so.6 due to outdated version packaged with julia 
# see https://discourse.julialang.org/t/glibcxx-3-4-26-not-found-in-rcall/29113/10?u=laborg
# and https://github.com/JuliaLang/julia/issues/34276
# e.g. for Ubuntu 19.10 64bit - match your locations accordingly!
# cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 $JULIA_HOME/lib/julia/
Pkg.build("RCall")