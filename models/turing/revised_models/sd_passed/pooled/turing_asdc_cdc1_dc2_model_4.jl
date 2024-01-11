using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_preDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_preDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_cDC2b::ContinuousUnivariateDistribution
    R_preDC::Float64
    R_precDC1bm::Float64
    R_precDC2bm::Float64
end

function Distributions.length(d::MyDistribution)
    8# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_preDCbm) #p_preDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    

    b[5] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1])) #Δ_cDC1bm
    b[6] = rand(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[5])) #Δ_cDC2bm
    b[7] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[5] -b[6])*d.R_preDC)) #Δ_cDC1b
    b[8] = rand(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (b[1] - b[5] -b[6]-b[7]/d.R_preDC)*d.R_preDC)) #Δ_cDC2b
    b[4] = rand(truncated(d.dδ_preDCb,0.0, -2e-12 +  (b[1] - b[5] -b[6]-b[7]/d.R_preDC-b[8]/d.R_preDC)*d.R_preDC)) #δ_preDCb
    
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_preDCbm) #p_preDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    
    x[5] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])) #Δ_cDC1bm
    x[6] = rand(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[5])) #Δ_cDC2bm
    x[7] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*d.R_preDC)) #Δ_cDC1b
    x[8] = rand(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/d.R_preDC)*d.R_preDC)) #Δ_cDC2b
    x[4] = rand(truncated(d.dδ_preDCb,0.0, -2e-12 +  (x[1] - x[5] -x[6]-x[7]/d.R_preDC-x[8]/d.R_preDC)*d.R_preDC)) #δ_preDCb
    
    return
end

function Distributions.rand(rng::Random.AbstractRNG,d::MyDistribution)
	x = zeros(length(d))
    Distributions._rand!(rng,d, x)
    return x
end



function Distributions._logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_preDCbm ,b[1]) #p_preDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
 
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[5]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[5]) ,b[6]) #Δ_cDC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[5] -b[6])*d.R_preDC), b[7]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (b[1] - b[5] -b[6]-b[7]/d.R_preDC)*d.R_preDC), b[8]) #Δ_cDC2b
    l += logpdf(truncated(d.dδ_preDCb,0.0, -2e-12 +  (b[1] - b[5] -b[6]-b[7]/d.R_preDC-b[8]/d.R_preDC)*d.R_preDC), b[4]) #δ_preDCb
  
    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_preDCbm ,b[1]) #p_preDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
 
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[5]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[5]) ,b[6]) #Δ_cDC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[5] -b[6])*d.R_preDC), b[7]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (b[1] - b[5] -b[6]-b[7]/d.R_preDC)*d.R_preDC), b[8]) #Δ_cDC2b
    l += logpdf(truncated(d.dδ_preDCb,0.0, -2e-12 +  (b[1] - b[5] -b[6]-b[7]/d.R_preDC-b[8]/d.R_preDC)*d.R_preDC), b[4]) #δ_preDCb
  
    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_preDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_preDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_cDC2b::ContinuousUnivariateDistribution
    R_preDC::Float64
    R_precDC1bm::Float64
    R_precDC2bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_preDCbm)(x[1]) #p_preDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_cDC2bm)(x[3]) #p_cDC2bm
    
    # lower_p_preDCbm = (λ_preDC + x[7] + x[8])
    # λ_preDC = (x[9] + x[10] + x[4]) / d.R_preDC
    
    y[5] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[5]) #Δ_cDC1bm
    y[6] = bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[5]) )(x[6]) #Δ_cDC2bm
    y[7] = bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*b.R_preDC))(x[7]) #Δ_cDC1b
    y[8] = bijector(truncated(b.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_preDC)*b.R_preDC))(x[8]) #Δ_cDC2b
    y[4] = bijector(truncated(b.dδ_preDCb,0.0,  -2e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_preDC-x[8]/b.R_preDC)*b.R_preDC))(x[4]) #δ_preDCb

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_preDCbm))(y[1]) #p_preDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_cDC2bm))(y[3]) #p_cDC2bm
    
    x[5] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[5]) #Δ_cDC1bm
    x[6] = inv(bijector(truncated(b.orig.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[5]) ))(y[6]) #Δ_cDC2bm
    x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*b.orig.R_preDC)))(y[7]) #Δ_cDC1b
    x[8] = inv(bijector(truncated(b.orig.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/b.orig.R_preDC)*b.orig.R_preDC)))(y[8]) #Δ_cDC2b
    x[4] = inv(bijector(truncated(b.orig.dδ_preDCb,0.0, -2e-12 +  (x[1] - x[5] -x[6]-x[7]/b.orig.R_preDC-x[8]/b.orig.R_preDC)*b.orig.R_preDC)))(y[4]) #δ_preDCb
    
    
    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_preDCbm),x[1]) #p_preDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_cDC2bm),x[3]) #p_cDC2bm
    
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])),x[5]) #Δ_cDC1bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[5])),x[6]) #Δ_cDC2bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*b.R_preDC)),x[7]) #Δ_cDC1b
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_preDC)*b.R_preDC)),x[8]) #Δ_cDC2b
    l += logabsdetjac(bijector(truncated(b.dδ_preDCb,0.0,  -2e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_preDC-x[8]/b.R_preDC)*b.R_preDC)),x[4]) #δ_preDCb

    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_preDCbm,d.dp_cDC1bm,d.dp_cDC2bm,d.dδ_preDCb,d.dΔ_cDC1bm,d.dΔ_cDC2bm,d.dΔ_cDC1b,d.dΔ_cDC2b,d.R_preDC,d.R_precDC1bm,d.R_precDC2bm)



function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat ,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode;save_idxs=[4,5,6], solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=save_idxs, trajectories=length(label_p), kwargs...)
end



@model function _turing_model(data::Array{Float64,1}, data_sd::Array{Float64,1}, metadata::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple; ode_parallel_mode=EnsembleSerial(), ode_args = (;))
    ### unpack R data
    @unpack R_preDC, R_cDC1, R_cDC2, R_precDC1bm, R_precDC2bm, R_precDC1b, R_precDC2b = metadata.R
    
    ### priors
    par ~ MyDistribution(priors.p_preDCbm, priors.p_cDC1bm, priors.p_cDC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),R_preDC, R_precDC1bm,R_precDC2bm)
    p_preDCbm, p_cDC1bm, p_cDC2bm, δ_preDCb, Δ_cDC1bm, Δ_cDC2bm, Δ_cDC1b, Δ_cDC2b = par           
    λ_preDC = (Δ_cDC1b + Δ_cDC2b + δ_preDCb) / R_preDC
    
    σ ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf),3)


    ### compound parameter
    δ_preDCbm = p_preDCbm .- λ_preDC .- Δ_cDC1bm .- Δ_cDC2bm
    δ_cDC1bm = p_cDC1bm .+ Δ_cDC1bm .* R_precDC1bm
    δ_cDC2bm = p_cDC2bm .+ Δ_cDC2bm .* R_precDC2bm
    δ_cDC1b = Δ_cDC1b .* R_precDC1b
    δ_cDC2b = Δ_cDC2b .* R_precDC2b

    theta = [[p_preDCbm, δ_preDCbm, p_cDC1bm, δ_cDC1bm, p_cDC2bm, δ_cDC2bm, δ_preDCb, δ_cDC1b, δ_cDC2b, λ_preDC, Δ_cDC1bm, Δ_cDC2bm, Δ_cDC1b, Δ_cDC2b] for j in 1:metadata.n_indv]
    
    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## hierachical error
    ### sd passed
    μ ~ MvNormal(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), σ[metadata.order.population])


    ## calculate likelihood
    data ~ MvNormal(μ, data_sd)

   

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf.(Normal.(μ, data_sd), data),
    parameters =(;p_preDCbm=p_preDCbm, δ_preDCbm=δ_preDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_cDC2bm=p_cDC2bm, δ_cDC2bm=δ_cDC2bm, δ_preDCb=δ_preDCb, δ_cDC1b=δ_cDC1b, δ_cDC2b=δ_cDC2b, λ_preDC=λ_preDC, Δ_cDC1bm=Δ_cDC1bm, Δ_cDC2bm=Δ_cDC2bm, Δ_cDC1b=Δ_cDC1b, Δ_cDC2b=Δ_cDC2b))
end


par_range = (;p_preDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_cDC2bm = (0.0,1.0),
δ_preDCb = (0.0,1.0),
Δ_cDC1bm = (0.0,2.0),
Δ_cDC2bm = (0.0,2.0),
Δ_cDC1b = (0.0,2.0),
Δ_cDC2b = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
