using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_DC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_DC2bm::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_DC2::ContinuousUnivariateDistribution
    RASDC::Float64
    RASDC_cDC1_bm::Float64
    RASDC_DC2_bm::Float64
end

function Distributions.length(d::MyDistribution)
    8# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(8)

    b[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_DC2bm) #p_DC2bm
    
    b[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1])) #Δ_cDC1bm
    b[8] = rand(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  b[1] - b[7])) #Δ_DC2bm
    b[4] = rand(truncated(d.dδ_ASDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8])*d.RASDC)) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.RASDC_cDC1_bm
    upper_λ_DC2 = b[3] + b[8] * d.RASDC_DC2_bm
    b[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    b[6] = rand(truncated(d.dλ_DC2, -Inf, upper_λ_DC2)) #λ_DC2 
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_DC2bm) #p_DC2bm
    
    x[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])) #Δ_cDC1bm
    x[8] = rand(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[7])) #Δ_DC2bm
    x[4] = rand(truncated(d.dδ_ASDCb,0.0, -4e-12 +  (x[1] - x[7] -x[8])*d.RASDC)) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * d.RASDC_cDC1_bm
    upper_λ_DC2 = x[3] + x[8] * d.RASDC_DC2_bm
    x[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    x[6] = rand(truncated(d.dλ_DC2, -Inf, upper_λ_DC2)) #λ_DC2 
    return
end

function Distributions.rand(rng::Random.AbstractRNG,d::MyDistribution)
	x = zeros(8)
    Distributions._rand!(rng,d, x)
    return x
end



function Distributions._logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_DC2bm ,b[3]) #p_DC2bm
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  b[1] - b[7]) ,b[8]) #Δ_DC2bm
    l += logpdf(truncated(d.dδ_ASDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8])*d.RASDC), b[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.RASDC_cDC1_bm
    upper_λ_DC2 = b[3] + b[8] * d.RASDC_DC2_bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_DC2, -Inf, upper_λ_DC2), b[6]) #λ_DC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_DC2bm ,b[3]) #p_DC2bm
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  b[1] - b[7]) ,b[8]) #Δ_DC2bm
    l += logpdf(truncated(d.dδ_ASDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8])*d.RASDC), b[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.RASDC_cDC1_bm
    upper_λ_DC2 = b[3] + b[8] * d.RASDC_DC2_bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_DC2, -Inf, upper_λ_DC2), b[6]) #λ_DC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_DC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_DC2bm::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_DC2::ContinuousUnivariateDistribution
    RASDC::Float64
    RASDC_cDC1_bm::Float64
    RASDC_DC2_bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_ASDCbm)(x[1]) #p_ASDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_DC2bm)(x[3]) #p_DC2bm
    
    y[7] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[7]) #Δ_cDC1bm
    y[8] = bijector(truncated(b.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[7]) )(x[8]) #Δ_DC2bm
    y[4] = bijector(truncated(b.dδ_ASDCb,0.0,  -2e-12 + (x[1] - x[7] -x[8])*b.RASDC))(x[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.RASDC_cDC1_bm
    upper_λ_DC2 = x[3] + x[8] * b.RASDC_DC2_bm
    y[5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[5]) #λ_cDC1
    y[6] = bijector(truncated(b.dλ_DC2, -Inf, upper_λ_DC2))(x[6]) #λ_DC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_ASDCbm))(y[1]) #p_ASDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_DC2bm))(y[3]) #p_DC2bm
    
    x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[7]) #Δ_cDC1bm
    x[8] = inv(bijector(truncated(b.orig.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[7]) ))(y[8]) #Δ_DC2bm
    x[4] = inv(bijector(truncated(b.orig.dδ_ASDCb,0.0, -4e-12 +  (x[1] - x[7] -x[8])*b.orig.RASDC)))(y[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.orig.RASDC_cDC1_bm
    upper_λ_DC2 = x[3] + x[8] * b.orig.RASDC_DC2_bm
    x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
    x[6] = inv(bijector(truncated(b.orig.dλ_DC2, -Inf, upper_λ_DC2)))(y[6]) #λ_DC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_ASDCbm),x[1]) #p_ASDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_DC2bm),x[3]) #p_DC2bm
    
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])),x[7]) #Δ_cDC1bm
    l += logabsdetjac(bijector(truncated(b.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[7])),x[8]) #Δ_DC2bm
    l += logabsdetjac(bijector(truncated(b.dδ_ASDCb,0.0,  -4e-12 + (x[1] - x[7] -x[8])*b.RASDC)),x[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.RASDC_cDC1_bm
    upper_λ_DC2 = x[3] + x[8] * b.RASDC_DC2_bm
    l += logabsdetjac(bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1)),x[5]) #λ_cDC1
    l += logabsdetjac(bijector(truncated(b.dλ_DC2, -Inf, upper_λ_DC2)),x[6]) #λ_DC2 


    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_ASDCbm,d.dp_cDC1bm,d.dp_DC2bm,d.dδ_ASDCb,d.dΔ_cDC1bm,d.dΔ_DC2bm,d.dλ_cDC1,d.dλ_DC2,d.RASDC,d.RASDC_cDC1_bm,d.RASDC_DC2_bm)



function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat, save_idxs=save_idxs,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode; solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=[4,5,6], trajectories=length(label_p), kwargs...)
end



@model function _turing_model(data::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple, sim::Bool=false, logp::Bool=false; ode_args = (;))
	### priors
    par ~ MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_DC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),data.R.RASDC, data.R.RASDC_cDC1_bm,data.R.RASDC_DC2_bm)
    p_ASDCbm, p_cDC1bm, p_DC2bm, δ_ASDCb, λ_cDC1, λ_DC2, Δ_cDC1bm, Δ_DC2bm = par           
    λ_ASDC = δ_ASDCb / data.R.RASDC
    
    σ ~ filldist(InverseGamma(2,3),3)


    ### compound parameter
    δ_ASDCbm = p_ASDCbm .- λ_ASDC .- Δ_cDC1bm .- Δ_DC2bm
    δ_cDC1bm = p_cDC1bm .+ Δ_cDC1bm .* data.R.RASDC_cDC1_bm .- λ_cDC1
    δ_DC2bm = p_DC2bm .+ Δ_DC2bm .* data.R.RASDC_DC2_bm .- λ_DC2
    δ_cDC1b = λ_cDC1 .* data.R.RcDC1
    δ_DC2b = λ_DC2 .* data.R.RDC2

    theta = [[p_ASDCbm, δ_ASDCbm, p_cDC1bm, δ_cDC1bm, p_DC2bm, δ_DC2bm, δ_ASDCb, δ_cDC1b, δ_DC2b, λ_ASDC, λ_cDC1, λ_DC2, Δ_cDC1bm, Δ_DC2bm] for j in 1:metadata.n_indv]

    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## calculate likelihood (mean)
    data ~ MvNormal(sol[metadata.order.donor][metadata.order.population, metadata.order.timepoint_idx], σ[metadata.order.population])

   

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf(MvNormal(sol[metadata.order.donor][metadata.order.population, metadata.order.time], σ[metadata.order.population]), data),
    parameters =(;p_ASDCbm=p_ASDCbm, δ_ASDCbm=δ_ASDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_DC2bm=p_DC2bm, δ_DC2bm=δ_DC2bm, δ_ASDCb=δ_ASDCb, δ_cDC1b=δ_cDC1b, δ_DC2b=δ_DC2b, λ_ASDC=λ_ASDC, λ_cDC1=λ_cDC1, λ_DC2=λ_DC2, Δ_cDC1bm=Δ_cDC1bm, Δ_DC2bm=Δ_DC2bm))
end


par_range = (;p_ASDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_DC2bm = (0.0,1.0),
δ_ASDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_DC2 = (0.0,2.0),
Δ_cDC1bm = (0.0,2.0),
Δ_DC2bm = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
