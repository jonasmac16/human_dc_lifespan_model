using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_cDC2b::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    R_ASDC::Float64
    R_precDC1bm::Float64
    R_precDC2bm::Float64
end

function Distributions.length(d::MyDistribution)
    10# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    b[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1])) #Δ_cDC1bm
    b[8] = rand(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[7])) #Δ_cDC2bm
    b[9] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[7] -b[8])*d.R_ASDC)) #Δ_cDC1b
    b[10] = rand(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (b[1] - b[7] -b[8]-b[9]/d.R_ASDC)*d.R_ASDC)) #Δ_cDC2b
    b[4] = rand(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (b[1] - b[7] -b[8]-b[9]/d.R_ASDC-b[10]/d.R_ASDC)*d.R_ASDC)) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.R_precDC1bm
    upper_λ_cDC2 = b[3] + b[8] * d.R_precDC2bm
    b[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    b[6] = rand(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2)) #λ_cDC2 
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::AbstractArray{Float64,1})
    x[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    x[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])) #Δ_cDC1bm
    x[8] = rand(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7])) #Δ_cDC2bm
    x[9] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*d.R_ASDC)) #Δ_cDC1b
    x[10] = rand(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[7] -x[8]-x[9]/d.R_ASDC)*d.R_ASDC)) #Δ_cDC2b
    x[4] = rand(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (x[1] - x[7] -x[8]-x[9]/d.R_ASDC-x[10]/d.R_ASDC)*d.R_ASDC)) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * d.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * d.R_precDC2bm
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
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[7]) ,b[8]) #Δ_cDC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[7] -b[8])*d.R_ASDC), b[9]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (b[1] - b[7] -b[8]-b[9]/d.R_ASDC)*d.R_ASDC), b[10]) #Δ_cDC2b
    l += logpdf(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (b[1] - b[7] -b[8]-b[9]/d.R_ASDC-b[10]/d.R_ASDC)*d.R_ASDC), b[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.R_precDC1bm
    upper_λ_cDC2 = b[3] + b[8] * d.R_precDC2bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -5e-12 +  b[1] - b[7]) ,b[8]) #Δ_cDC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[7] -b[8])*d.R_ASDC), b[9]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_cDC2b,0.0, -3e-12 + (b[1] - b[7] -b[8]-b[9]/d.R_ASDC)*d.R_ASDC), b[10]) #Δ_cDC2b
    l += logpdf(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (b[1] - b[7] -b[8]-b[9]/d.R_ASDC-b[10]/d.R_ASDC)*d.R_ASDC), b[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = b[2] + b[7] * d.R_precDC1bm
    upper_λ_cDC2 = b[3] + b[8] * d.R_precDC2bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_cDC2b::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    R_ASDC::Float64
    R_precDC1bm::Float64
    R_precDC2bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_ASDCbm)(x[1]) #p_ASDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_cDC2bm)(x[3]) #p_cDC2bm
    
    y[7] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[7]) #Δ_cDC1bm
    y[8] = bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7]) )(x[8]) #Δ_cDC2bm
    y[9] = bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.R_ASDC))(x[9]) #Δ_cDC1b
    y[10] = bijector(truncated(b.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[7] -x[8]-x[9]/b.R_ASDC)*b.R_ASDC))(x[10]) #Δ_cDC2b
    y[4] = bijector(truncated(b.dδ_ASDCb,0.0,  -2e-12 + (x[1] - x[7] -x[8]-x[9]/b.R_ASDC-x[10]/b.R_ASDC)*b.R_ASDC))(x[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * b.R_precDC2bm
    y[5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[5]) #λ_cDC1
    y[6] = bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2))(x[6]) #λ_cDC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_ASDCbm))(y[1]) #p_ASDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_cDC2bm))(y[3]) #p_cDC2bm
    
    x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[7]) #Δ_cDC1bm
    x[8] = inv(bijector(truncated(b.orig.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7]) ))(y[8]) #Δ_cDC2bm
    x[9] = inv(bijector(truncated(b.orig.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.orig.R_ASDC)))(y[9]) #Δ_cDC1b
    x[10] = inv(bijector(truncated(b.orig.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[7] -x[8]-x[9]/b.orig.R_ASDC)*b.orig.R_ASDC)))(y[10]) #Δ_cDC2b
    x[4] = inv(bijector(truncated(b.orig.dδ_ASDCb,0.0, -2e-12 +  (x[1] - x[7] -x[8]-x[9]/b.orig.R_ASDC-x[10]/b.orig.R_ASDC)*b.orig.R_ASDC)))(y[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.orig.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * b.orig.R_precDC2bm
    x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
    x[6] = inv(bijector(truncated(b.orig.dλ_cDC2, -Inf, upper_λ_cDC2)))(y[6]) #λ_cDC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_ASDCbm),x[1]) #p_ASDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_cDC2bm),x[3]) #p_cDC2bm
    
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])),x[7]) #Δ_cDC1bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7])),x[8]) #Δ_cDC2bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.R_ASDC)),x[9]) #Δ_cDC1b
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[7] -x[8]-x[9]/b.R_ASDC)*b.R_ASDC)),x[10]) #Δ_cDC2b
    l += logabsdetjac(bijector(truncated(b.dδ_ASDCb,0.0,  -2e-12 + (x[1] - x[7] -x[8]-x[9]/b.R_ASDC-x[10]/b.R_ASDC)*b.R_ASDC)),x[4]) #δ_ASDCb
    
    
    upper_λ_cDC1 = x[2] + x[7] * b.R_precDC1bm
    upper_λ_cDC2 = x[3] + x[8] * b.R_precDC2bm
    l += logabsdetjac(bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1)),x[5]) #λ_cDC1
    l += logabsdetjac(bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2)),x[6]) #λ_cDC2 


    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_ASDCbm,d.dp_cDC1bm,d.dp_cDC2bm,d.dδ_ASDCb,d.dΔ_cDC1bm,d.dΔ_cDC2bm,d.dΔ_cDC1b,d.dΔ_cDC2b,d.dλ_cDC1,d.dλ_cDC2,d.R_ASDC,d.R_precDC1bm,d.R_precDC2bm)

# function (b::MyBijector)(x::AbstractMatrix)
# 	y = similar(x)

#     y[:,1] = bijector(b.dp_ASDCbm)(x[:,1]) #p_ASDCbm
#     y[:,2] = bijector(b.dp_cDC1bm)(x[:,2]) #p_cDC1bm
#     y[:,3] = bijector(b.dp_cDC2bm)(x[:,3]) #p_cDC2bm
    
#     y[:,7] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[:,7]) #Δ_cDC1bm
#     y[:,8] = bijector(truncated(b.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7]) )(x[:,8]) #Δ_cDC2bm
#     y[:,9] = bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.R_ASDC))(x[:,9]) #Δ_cDC1b
#     y[:,10] = bijector(truncated(b.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[7] -x[8]-x[9]/b.R_ASDC)*b.R_ASDC))(x[:,10]) #Δ_cDC2b
#     y[:,4] = bijector(truncated(b.dδ_ASDCb,0.0,  -2e-12 + (x[1] - x[7] -x[8]-x[9]/b.R_ASDC-x[10]/b.R_ASDC)*b.R_ASDC))(x[:,4]) #δ_ASDCb
    
    
#     upper_λ_cDC1 = x[2] + x[7] * b.R_precDC1bm
#     upper_λ_cDC2 = x[3] + x[8] * b.R_precDC2bm
#     y[:,5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[:,5]) #λ_cDC1
#     y[:,6] = bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2))(x[:,6]) #λ_cDC2 

#     return y
# end
# function (b::Inverse{<:MyBijector})(y::AbstractMatrix)
# 	x = similar(y)

#     x[1] = inv(bijector(b.orig.dp_ASDCbm))(y[1]) #p_ASDCbm
#     x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
#     x[3] = inv(bijector(b.orig.dp_cDC2bm))(y[3]) #p_cDC2bm
    
#     x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[7]) #Δ_cDC1bm
#     x[8] = inv(bijector(truncated(b.orig.dΔ_cDC2bm, 0.0, -5e-12 +  x[1] - x[7]) ))(y[8]) #Δ_cDC2bm
#     x[9] = inv(bijector(truncated(b.orig.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.orig.R_ASDC)))(y[9]) #Δ_cDC1b
#     x[10] = inv(bijector(truncated(b.orig.dΔ_cDC2b,0.0, -3e-12 + (x[1] - x[7] -x[8]-x[9]/b.orig.R_ASDC)*b.orig.R_ASDC)))(y[10]) #Δ_cDC2b
#     x[4] = inv(bijector(truncated(b.orig.dδ_ASDCb,0.0, -2e-12 +  (x[1] - x[7] -x[8]-x[9]/b.orig.R_ASDC-x[10]/b.orig.R_ASDC)*b.orig.R_ASDC)))(y[4]) #δ_ASDCb
    
    
#     upper_λ_cDC1 = x[2] + x[7] * b.orig.R_precDC1bm
#     upper_λ_cDC2 = x[3] + x[8] * b.orig.R_precDC2bm
#     x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
#     x[6] = inv(bijector(truncated(b.orig.dλ_cDC2, -Inf, upper_λ_cDC2)))(y[6]) #λ_cDC2 

#     return x
# end




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



@model function _turing_model(data::Array{Float64,1}, metadata::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple, ::Type{T} = Float64; ode_parallel_mode=EnsembleSerial(), ode_args = (;)) where {T <: Real}
    ### unpack R data
    @unpack R_ASDC, R_cDC1, R_cDC2, R_precDC1bm, R_precDC2bm, R_precDC1b, R_precDC2b = metadata.R
    
    ### priors
    prior_dist = MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_cDC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), R_ASDC, R_precDC1bm, R_precDC2bm)

    par = Vector{Array{T,1}}(undef, metadata.n_indv)
    for j in 1:metadata.n_indv
        par[j] ~ prior_dist
    end
        
    p_ASDCbm, p_cDC1bm, p_cDC2bm, δ_ASDCb, λ_cDC1, λ_cDC2, Δ_cDC1bm, Δ_cDC2bm, Δ_cDC1b, Δ_cDC2b = assign_par(par, length(prior_dist), metadata.n_indv)

    # p_ASDCbm[j], p_cDC1bm[j], p_cDC2bm[j], δ_ASDCb[j], λ_cDC1[j], λ_cDC2[j], Δ_cDC1bm[j], Δ_cDC2bm[j], Δ_cDC1b[j], Δ_cDC2b[j]
    # p_ASDCbm, p_cDC1bm, p_cDC2bm, δ_ASDCb, λ_cDC1, λ_cDC2, Δ_cDC1bm, Δ_cDC2bm, Δ_cDC1b, Δ_cDC2b assign_par(par, length(prior_dist), metadata.n_indv)
    
    λ_ASDC = (Δ_cDC1b .+ Δ_cDC2b .+ δ_ASDCb) ./ R_ASDC
    
    σ ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf),3, metadata.n_indv)
    ν ~ filldist(LogNormal(2.0, 1.0),3, metadata.n_indv)


    ### compound parameter
    δ_ASDCbm = p_ASDCbm .- λ_ASDC .- Δ_cDC1bm .- Δ_cDC2bm
    δ_cDC1bm = p_cDC1bm .+ Δ_cDC1bm .* R_precDC1bm .- λ_cDC1
    δ_cDC2bm = p_cDC2bm .+ Δ_cDC2bm .* R_precDC2bm .- λ_cDC2
    δ_cDC1b = λ_cDC1 .* R_cDC1 .+ Δ_cDC1b .* R_precDC1b
    δ_cDC2b = λ_cDC2 .* R_cDC2 .+ Δ_cDC2b .* R_precDC2b

    ## parameter vector
    theta = [[p_ASDCbm[j], δ_ASDCbm[j], p_cDC1bm[j], δ_cDC1bm[j], p_cDC2bm[j], δ_cDC2bm[j], δ_ASDCb[j], δ_cDC1b[j], δ_cDC2b[j], λ_ASDC[j], λ_cDC1[j], λ_cDC2[j], Δ_cDC1bm[j], Δ_cDC2bm[j], Δ_cDC1b[j], Δ_cDC2b[j]] for j in 1:metadata.n_indv]


    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## calculate likelihood (mean)
    # data ~ MvNormal(sol[metadata.order.donor][metadata.order.population, metadata.order.time], σ[metadata.order.population, metadata.order.donor])
    data ~ arraydist(LocationScale.(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), map(j -> σ[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor)), TDist.(map(j -> ν[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor)))))



   

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf.(LocationScale.(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), map(j -> σ[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor)), TDist.(map(j -> ν[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor)))), data),
    parameters =(;p_ASDCbm=p_ASDCbm, δ_ASDCbm=δ_ASDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_cDC2bm=p_cDC2bm, δ_cDC2bm=δ_cDC2bm, δ_ASDCb=δ_ASDCb, δ_cDC1b=δ_cDC1b, δ_cDC2b=δ_cDC2b, λ_ASDC=λ_ASDC, λ_cDC1=λ_cDC1, λ_cDC2=λ_cDC2, Δ_cDC1bm=Δ_cDC1bm, Δ_cDC2bm=Δ_cDC2bm, Δ_cDC1b=Δ_cDC1b, Δ_cDC2b=Δ_cDC2b))
end


par_range = (;p_ASDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_cDC2bm = (0.0,1.0),
δ_ASDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_cDC2 = (0.0,2.0),
Δ_cDC1bm = (0.0,2.0),
Δ_cDC2bm = (0.0,2.0),
Δ_cDC1b = (0.0,2.0),
Δ_cDC2b = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0),
ν1 = (0.0,2.0),
ν2 = (0.0,2.0),
ν3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
