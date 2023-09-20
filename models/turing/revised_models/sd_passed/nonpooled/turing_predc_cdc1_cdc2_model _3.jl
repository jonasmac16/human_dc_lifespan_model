using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    RASDC::Float64
    RASDC_cDC1_bm::Float64
    RASDC_cDC2_bm::Float64
end

function Distributions.length(d::MyDistribution)
    6# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    b[4] = rand(truncated(d.dδ_ASDCb,0.0, b[1]*d.RASDC)) #δ_ASDCb
        
    upper_λ_cDC1 = b[2]
    upper_λ_cDC2 = b[3]
    b[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    b[6] = rand(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2)) #λ_cDC2 
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    x[4] = rand(truncated(d.dδ_ASDCb,0.0, x[1]*d.RASDC)) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3]
    x[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    x[6] = rand(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2)) #λ_cDC2 
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
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
    l += logpdf(truncated(d.dδ_ASDCb,0.0, b[1]*d.RASDC), b[4]) #δ_ASDCb
        
    upper_λ_cDC1 = b[2]
    upper_λ_cDC2 = b[3]
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    

    l += logpdf(truncated(d.dδ_ASDCb,0.0, b[1] *d.RASDC), b[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2]
    upper_λ_cDC2 = b[3]
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    RASDC::Float64
    RASDC_cDC1_bm::Float64
    RASDC_cDC2_bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_ASDCbm)(x[1]) #p_ASDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_cDC2bm)(x[3]) #p_cDC2bm
    
    y[4] = bijector(truncated(b.dδ_ASDCb,0.0,  x[1]*b.RASDC))(x[4]) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3]
    y[5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[5]) #λ_cDC1
    y[6] = bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2))(x[6]) #λ_cDC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_ASDCbm))(y[1]) #p_ASDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_cDC2bm))(y[3]) #p_cDC2bm
    
    x[4] = inv(bijector(truncated(b.orig.dδ_ASDCb,0.0, x[1]*b.orig.RASDC)))(y[4]) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3]
    x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
    x[6] = inv(bijector(truncated(b.orig.dλ_cDC2, -Inf, upper_λ_cDC2)))(y[6]) #λ_cDC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_ASDCbm),x[1]) #p_ASDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_cDC2bm),x[3]) #p_cDC2bm
    
    l += logabsdetjac(bijector(truncated(b.dδ_ASDCb,0.0,  x[1]*b.RASDC)),x[4]) #δ_ASDCb
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3]
    l += logabsdetjac(bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1)),x[5]) #λ_cDC1
    l += logabsdetjac(bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2)),x[6]) #λ_cDC2 


    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_ASDCbm,d.dp_cDC1bm,d.dp_cDC2bm,d.dδ_ASDCb,d.dλ_cDC1,d.dλ_cDC2,d.RASDC,d.RASDC_cDC1_bm,d.RASDC_cDC2_bm)



function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat, save_idxs=save_idxs,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode; solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=[4,5,6], trajectories=length(label_p), kwargs...)
end



@model function _turing_model(data::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple, sim::Bool=false, logp::Bool=false; ode_args = (;))
	### priors
    par ~ MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_cDC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),data.R.RASDC, data.R.RASDC_cDC1_bm,data.R.RASDC_cDC2_bm)
    p_ASDCbm, p_cDC1bm, p_cDC2bm, δ_ASDCb, λ_cDC1, λ_cDC2 = eachrow(par)           
    λ_ASDC = δ_ASDCb ./ data.R.RASDC
    
    σ1 ~ TruncatedNormal(0.0, 1.0, 0.0,Inf)
    σ2 ~ TruncatedNormal(0.0, 1.0, 0.0,Inf)
    σ3 ~ TruncatedNormal(0.0, 1.0, 0.0,Inf)


    ### compound parameter
    δ_ASDCbm = p_ASDCbm .- λ_ASDC
    δ_cDC1bm = p_cDC1bm .- λ_cDC1
    δ_cDC2bm = p_cDC2bm .- λ_cDC2
    δ_cDC1b = λ_cDC1 .* data.R.RcDC1
    δ_cDC2b = λ_cDC2 .* data.R.RcDC2

    theta = [[p_ASDCbm[j], δ_ASDCbm[j], p_cDC1bm[j], δ_cDC1bm[j], p_cDC2bm[j], δ_cDC2bm[j], δ_ASDCb[j], δ_cDC1b[j], δ_cDC2b[j], λ_ASDC[j], λ_cDC1[j], λ_cDC2[j]] for j in 1:metadata.n_indv]
    
    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## hierachical error
    ### sd passed
    μ ~ MvNormal(sol[metadata.order.donor][metadata.order.population, metadata.order.timepoint_idx], σ[metadata.order.population, metadata.order.donor])


    ## calculate likelihood
    data ~ MvNormal(μ, data_sd)

   

    

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf(MvNormal(μ, data_sd), data),
    parameters =(;p_ASDCbm=p_ASDCbm, δ_ASDCbm=δ_ASDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_cDC2bm=p_cDC2bm, δ_cDC2bm=δ_cDC2bm, δ_ASDCb=δ_ASDCb, δ_cDC1b=δ_cDC1b, δ_cDC2b=δ_cDC2b, λ_ASDC=λ_ASDC, λ_cDC1=λ_cDC1, λ_cDC2=λ_cDC2, Δ_cDC1bm=Δ_cDC1bm, Δ_cDC2bm=Δ_cDC2bm, Δ_cDC1b=Δ_cDC1b, Δ_cDC2b=Δ_cDC2b))
end


par_range = (;p_ASDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_cDC2bm = (0.0,1.0),
δ_ASDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_cDC2 = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
