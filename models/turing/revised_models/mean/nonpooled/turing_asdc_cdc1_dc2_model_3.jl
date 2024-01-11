using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_DC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_DC2::ContinuousUnivariateDistribution
    R_ASDC::Float64
    R_ASDCcDC1bm::Float64
    R_ASDCDC2bm::Float64
end

function Distributions.length(d::MyDistribution)
    6# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_DC2bm) #p_DC2bm
    
    b[4] = rand(truncated(d.dδ_ASDCb,0.0, b[1]*d.R_ASDC)) #δ_ASDCb
        
    upper_λ_cDC1 = b[2]
    upper_λ_DC2 = b[3]
    b[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    b[6] = rand(truncated(d.dλ_DC2, -Inf, upper_λ_DC2)) #λ_DC2 
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_DC2bm) #p_DC2bm
    
    x[4] = rand(truncated(d.dδ_ASDCb,0.0, x[1]*d.R_ASDC)) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_DC2 = x[3]
    x[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    x[6] = rand(truncated(d.dλ_DC2, -Inf, upper_λ_DC2)) #λ_DC2 
    return
end

function Distributions.rand(rng::Random.AbstractRNG,d::MyDistribution)
	x = zeros(length(d))
    Distributions._rand!(rng,d, x)
    return x
end



function Distributions._logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_DC2bm ,b[3]) #p_DC2bm
    
    l += logpdf(truncated(d.dδ_ASDCb,0.0, b[1]*d.R_ASDC), b[4]) #δ_ASDCb
        
    upper_λ_cDC1 = b[2]
    upper_λ_DC2 = b[3]
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_DC2, -Inf, upper_λ_DC2), b[6]) #λ_DC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_DC2bm ,b[3]) #p_DC2bm
    

    l += logpdf(truncated(d.dδ_ASDCb,0.0, b[1] *d.R_ASDC), b[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2]
    upper_λ_DC2 = b[3]
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_DC2, -Inf, upper_λ_DC2), b[6]) #λ_DC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_DC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_DC2::ContinuousUnivariateDistribution
    R_ASDC::Float64
    R_ASDCcDC1bm::Float64
    R_ASDCDC2bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_ASDCbm)(x[1]) #p_ASDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_DC2bm)(x[3]) #p_DC2bm
    
    y[4] = bijector(truncated(b.dδ_ASDCb,0.0,  x[1]*b.R_ASDC))(x[4]) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_DC2 = x[3]
    y[5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[5]) #λ_cDC1
    y[6] = bijector(truncated(b.dλ_DC2, -Inf, upper_λ_DC2))(x[6]) #λ_DC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_ASDCbm))(y[1]) #p_ASDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_DC2bm))(y[3]) #p_DC2bm
    
    x[4] = inv(bijector(truncated(b.orig.dδ_ASDCb,0.0, x[1]*b.orig.R_ASDC)))(y[4]) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_DC2 = x[3]
    x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
    x[6] = inv(bijector(truncated(b.orig.dλ_DC2, -Inf, upper_λ_DC2)))(y[6]) #λ_DC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_ASDCbm),x[1]) #p_ASDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_DC2bm),x[3]) #p_DC2bm
    
    l += logabsdetjac(bijector(truncated(b.dδ_ASDCb,0.0,  x[1]*b.R_ASDC)),x[4]) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_DC2 = x[3]
    l += logabsdetjac(bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1)),x[5]) #λ_cDC1
    l += logabsdetjac(bijector(truncated(b.dλ_DC2, -Inf, upper_λ_DC2)),x[6]) #λ_DC2 


    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_ASDCbm,d.dp_cDC1bm,d.dp_DC2bm,d.dδ_ASDCb,d.dλ_cDC1,d.dλ_DC2,d.R_ASDC,d.R_ASDCcDC1bm,d.R_ASDCDC2bm)



function assign_par(x::AbstractArray, npar::Int, nrep::Int)
    return [[x[k][j] for k in 1:nrep] for j in 1:npar]
end

function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat ,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode;save_idxs=[4,5,6], solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=save_idxs, trajectories=length(label_p), kwargs...)
end



@model function _turing_model(data::Array{Float64,1}, metadata::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple, ::Type{T} = Float64; ode_parallel_mode=EnsembleSerial(), ode_args = (;)) where {T}
    ### unpack R data
    @unpack R_ASDC, R_cDC1, R_DC2, R_ASDCcDC1bm, R_ASDCDC2bm, R_ASDCcDC1b, R_ASDCDC2b = metadata.R
    
    ### priors
    prior_dist = MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_DC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),R_ASDC, R_ASDCcDC1bm,R_ASDCDC2bm)
    
    par = Vector{Array{T,1}}(undef, metadata.n_indv)
    for j in 1:metadata.n_indv
        par[j] ~ prior_dist
    end
        
    p_ASDCbm, p_cDC1bm, p_DC2bm, δ_ASDCb, λ_cDC1, λ_DC2 = assign_par(par, length(prior_dist), metadata.n_indv)          
    λ_ASDC = δ_ASDCb ./ R_ASDC
    
    σ ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf),3, metadata.n_indv)


    ### compound parameter
    δ_ASDCbm = p_ASDCbm .- λ_ASDC
    δ_cDC1bm = p_cDC1bm .- λ_cDC1
    δ_DC2bm = p_DC2bm .- λ_DC2
    δ_cDC1b = λ_cDC1 .* R_cDC1
    δ_DC2b = λ_DC2 .* R_DC2

    theta = [[p_ASDCbm[j], δ_ASDCbm[j], p_cDC1bm[j], δ_cDC1bm[j], p_DC2bm[j], δ_DC2bm[j], δ_ASDCb[j], δ_cDC1b[j], δ_DC2b[j], λ_ASDC[j], λ_cDC1[j], λ_DC2[j]] for j in 1:metadata.n_indv]
    
    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## calculate likelihood (mean)
    data ~ MvNormal(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), map(j -> σ[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor)))

    

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf.(Normal.(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), map(j -> σ[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor))), data),
    parameters =(;p_ASDCbm=p_ASDCbm, δ_ASDCbm=δ_ASDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_DC2bm=p_DC2bm, δ_DC2bm=δ_DC2bm, δ_ASDCb=δ_ASDCb, δ_cDC1b=δ_cDC1b, δ_DC2b=δ_DC2b, λ_ASDC=λ_ASDC, λ_cDC1=λ_cDC1, λ_DC2=λ_DC2))
end


par_range = (;p_ASDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_DC2bm = (0.0,1.0),
δ_ASDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_DC2 = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
