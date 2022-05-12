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
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    R_preDC::Float64
    R_precDC1bm::Float64
    R_precDC2bm::Float64
end

function Distributions.length(d::MyDistribution)
    8# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(8)

    b[1] = rand(d.dp_preDCbm) #p_preDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    b[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1])) #Δ_cDC1bm
    b[8] = rand(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[7])) #Δ_cDC2bm
    b[4] = rand(truncated(d.dδ_preDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8])*d.R_preDC)) #δ_preDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.R_precDC1bm
    upper_λ_cDC2 = b[3] + b[8] * d.R_precDC2bm
    b[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    b[6] = rand(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2)) #λ_cDC2 
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_preDCbm) #p_preDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    x[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])) #Δ_cDC1bm
    x[8] = rand(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7])) #Δ_cDC2bm
    x[4] = rand(truncated(d.dδ_preDCb,0.0, -4e-12 +  (x[1] - x[7] -x[8])*d.R_preDC)) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * d.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * d.R_precDC2bm
    x[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    x[6] = rand(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2)) #λ_cDC2 
    return
end

function Distributions.rand(rng::Random.AbstractRNG,d::MyDistribution)
	x = zeros(8)
    Distributions._rand!(rng,d, x)
    return x
end



function Distributions._logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_preDCbm ,b[1]) #p_preDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[7]) ,b[8]) #Δ_cDC2bm
    l += logpdf(truncated(d.dδ_preDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8])*d.R_preDC), b[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.R_precDC1bm
    upper_λ_cDC2 = b[3] + b[8] * d.R_precDC2bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_preDCbm ,b[1]) #p_preDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[7]) ,b[8]) #Δ_cDC2bm
    l += logpdf(truncated(d.dδ_preDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8])*d.R_preDC), b[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.R_precDC1bm
    upper_λ_cDC2 = b[3] + b[8] * d.R_precDC2bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_preDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_preDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    R_preDC::Float64
    R_precDC1bm::Float64
    R_precDC2bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_preDCbm)(x[1]) #p_preDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_cDC2bm)(x[3]) #p_cDC2bm
    
    y[7] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[7]) #Δ_cDC1bm
    y[8] = bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7]) )(x[8]) #Δ_cDC2bm
    y[4] = bijector(truncated(b.dδ_preDCb,0.0,  -2e-12 + (x[1] - x[7] -x[8])*b.R_preDC))(x[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * b.R_precDC2bm
    y[5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[5]) #λ_cDC1
    y[6] = bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2))(x[6]) #λ_cDC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_preDCbm))(y[1]) #p_preDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_cDC2bm))(y[3]) #p_cDC2bm
    
    x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[7]) #Δ_cDC1bm
    x[8] = inv(bijector(truncated(b.orig.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7]) ))(y[8]) #Δ_cDC2bm
    x[4] = inv(bijector(truncated(b.orig.dδ_preDCb,0.0, -4e-12 +  (x[1] - x[7] -x[8])*b.orig.R_preDC)))(y[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.orig.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * b.orig.R_precDC2bm
    x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
    x[6] = inv(bijector(truncated(b.orig.dλ_cDC2, -Inf, upper_λ_cDC2)))(y[6]) #λ_cDC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_preDCbm),x[1]) #p_preDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_cDC2bm),x[3]) #p_cDC2bm
    
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])),x[7]) #Δ_cDC1bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7])),x[8]) #Δ_cDC2bm
    l += logabsdetjac(bijector(truncated(b.dδ_preDCb,0.0,  -4e-12 + (x[1] - x[7] -x[8])*b.R_preDC)),x[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * b.R_precDC2bm
    l += logabsdetjac(bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1)),x[5]) #λ_cDC1
    l += logabsdetjac(bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2)),x[6]) #λ_cDC2 


    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_preDCbm,d.dp_cDC1bm,d.dp_cDC2bm,d.dδ_preDCb,d.dΔ_cDC1bm,d.dΔ_cDC2bm,d.dλ_cDC1,d.dλ_cDC2,d.R_preDC,d.R_precDC1bm,d.R_precDC2bm)



function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat ,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode;save_idxs=[4,5,6], solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=save_idxs, trajectories=length(label_p), kwargs...)
end

# function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Float64,1}, timepoints::Array{Float64,1};save_idxs=[4,5,6], solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
#     tmp_prob = prob_func(ODEprob, theta, label_p, timepoints)
#     return DifferentialEquations.solve(tmp_prob,solver; save_idxs=save_idxs, trajectories=length(label_p), kwargs...)
# end


@model function _turing_model(data::Array{Float64,1}, metadata::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple; ode_parallel_mode=EnsembleSerial(), ode_args = (;))
    ### unpack R data
    @unpack R_preDC, R_cDC1, R_cDC2, R_precDC1bm, R_precDC2bm, R_precDC1b, R_precDC2b = metadata.R
    
    ### priors
    par ~ MyDistribution(priors.p_preDCbm, priors.p_cDC1bm, priors.p_cDC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),R_preDC, R_precDC1bm,R_precDC2bm)
    p_preDCbm, p_cDC1bm, p_cDC2bm, δ_preDCb, λ_cDC1, λ_cDC2, Δ_cDC1bm, Δ_cDC2bm = par           

    σ ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf),3)


    ### compound parameter
    λ_preDC = δ_preDCb / R_preDC
    δ_preDCbm = p_preDCbm .- λ_preDC .- Δ_cDC1bm .- Δ_cDC2bm
    δ_cDC1bm = p_cDC1bm .+ Δ_cDC1bm .* R_precDC1bm .- λ_cDC1
    δ_cDC2bm = p_cDC2bm .+ Δ_cDC2bm .* R_precDC2bm .- λ_cDC2
    δ_cDC1b = λ_cDC1 .* R_cDC1
    δ_cDC2b = λ_cDC2 .* R_cDC2

    theta = [[p_preDCbm, δ_preDCbm, p_cDC1bm, δ_cDC1bm, p_cDC2bm, δ_cDC2bm, δ_preDCb, δ_cDC1b, δ_cDC2b, λ_preDC, λ_cDC1, λ_cDC2, Δ_cDC1bm, Δ_cDC2bm] for j in 1:metadata.n_indv]
    
    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## calculate likelihood (mean)
    data ~ MvNormal(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), σ[metadata.order.population])
    
   

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf.(Normal.(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), σ[metadata.order.population]), data),
    parameters =(;p_preDCbm=p_preDCbm, δ_preDCbm=δ_preDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_cDC2bm=p_cDC2bm, δ_cDC2bm=δ_cDC2bm, δ_preDCb=δ_preDCb, δ_cDC1b=δ_cDC1b, δ_cDC2b=δ_cDC2b, λ_preDC=λ_preDC, λ_cDC1=λ_cDC1, λ_cDC2=λ_cDC2, Δ_cDC1bm=Δ_cDC1bm, Δ_cDC2bm=Δ_cDC2bm))
end


par_range = (;p_preDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_cDC2bm = (0.0,1.0),
δ_preDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_cDC2 = (0.0,2.0),
Δ_cDC1bm = (0.0,2.0),
Δ_cDC2bm = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
