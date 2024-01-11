using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_DC3bm::ContinuousUnivariateDistribution
    dλ_DC3::ContinuousUnivariateDistribution
end

function Distributions.length(d::MyDistribution)
    2# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_DC3bm) #p_ASDCbm
    b[2] = rand(truncated(d.dλ_DC3, -Inf, b[1])) #λ_DC3

    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_DC3bm) #p_ASDCbm

    x[2] = rand(truncated(d.dλ_DC3, -Inf, x[1])) #λ_DC2 
    return
end

function Distributions.rand(rng::Random.AbstractRNG,d::MyDistribution)
	x = zeros(length(d))
    Distributions._rand!(rng,d, x)
    return x
end



function Distributions._logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_DC3bm ,b[1]) #p_ASDCbm
  
    l += logpdf(truncated(d.dλ_DC3, -Inf, b[1]), b[2]) #λ_DC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_DC3bm ,b[1]) #p_ASDCbm

    l += logpdf(truncated(d.dλ_DC3, -Inf, b[1]), b[2]) #λ_DC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_DC3bm::ContinuousUnivariateDistribution
    dλ_DC3::ContinuousUnivariateDistribution
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_DC3bm)(x[1]) #p_ASDCbm

    y[2] = bijector(truncated(b.dλ_DC3, -Inf, x[1]))(x[2]) #λ_DC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_DC3bm))(y[1]) #p_ASDCbm
    x[2] = inv(bijector(truncated(b.orig.dλ_DC3, -Inf, x[1])))(y[2]) #λ_DC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_DC3bm),x[1]) #p_ASDCbm
    l += logabsdetjac(bijector(truncated(b.dλ_DC3, -Inf, x[1])),x[2]) #λ_DC2 

    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_DC3bm,d.dλ_DC3)

function assign_par(x::AbstractArray, npar::Int, nrep::Int)
    return [[x[k][j] for k in 1:nrep] for j in 1:npar]
end

function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat ,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.DDEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode;save_idxs=[2], solver = MethodOfSteps(Vern6()),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=save_idxs, trajectories=length(label_p), kwargs...)
end

@model function _turing_model(data::Array{Float64,1}, metadata::NamedTuple, ode_prob::DDEProblem, solver, priors::NamedTuple, ::Type{T} = Float64; ode_parallel_mode=EnsembleSerial(), ode_args = (;)) where {T}
    ### unpack R data
    @unpack R_pDC = metadata.R
    
    ### priors
    prior_dist = MyDistribution(priors.p_DC3bm, Uniform(0.0,2.0))

    par = Vector{Array{T,1}}(undef, metadata.n_indv)
    for j in 1:metadata.n_indv
        par[j] ~ prior_dist
    end

    p_DC3bm, λ_DC3 = assign_par(par, length(prior_dist), metadata.n_indv)           
    tau ~ filldist(Uniform(0.0,10.0),metadata.n_indv)

    σ ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf),metadata.n_indv)

    ### compound parameter
    δ_DC3bm = p_DC3bm .- λ_DC3
    δ_DC3b = λ_DC3 .* R_pDC
    
    theta = [[p_DC3bm[j], δ_DC3bm[j], δ_DC3b[j], λ_DC3[j], tau[j]] for j in 1:metadata.n_indv]

    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)

    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    data ~ MvNormal(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), σ[metadata.order.donor])

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf.(Normal.(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), σ[metadata.order.population]), data),
    parameters =(;p_DC3bm=p_DC3bm, δ_DC3bm=δ_DC3bm, δ_DC3b=δ_DC3b, λ_DC3=λ_DC3, tau=tau))
end


par_range = (;p_DC3bm = (0.0,1.0),
λ_DC3 = (0.0,2.0),
tau = (0.0,10.0),
σ1 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
