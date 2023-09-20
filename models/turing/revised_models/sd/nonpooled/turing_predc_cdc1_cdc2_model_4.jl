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
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_DC2b::ContinuousUnivariateDistribution
    R_ASDC::Float64
    R_precDC1bm::Float64
    R_preDC2bm::Float64
end

function Distributions.length(d::MyDistribution)
    8# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_DC2bm) #p_DC2bm
    

    b[5] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1])) #Δ_cDC1bm
    b[6] = rand(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  b[1] - b[5])) #Δ_DC2bm
    b[7] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[5] -b[6])*d.R_ASDC)) #Δ_cDC1b
    b[8] = rand(truncated(d.dΔ_DC2b,0.0, -3e-12 + (b[1] - b[5] -b[6]-b[7]/d.R_ASDC)*d.R_ASDC)) #Δ_DC2b
    b[4] = rand(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (b[1] - b[5] -b[6]-b[7]/d.R_ASDC-b[8]/d.R_ASDC)*d.R_ASDC)) #δ_ASDCb
    
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_ASDCbm) #p_ASDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_DC2bm) #p_DC2bm
    
    
    x[5] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])) #Δ_cDC1bm
    x[6] = rand(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[5])) #Δ_DC2bm
    x[7] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*d.R_ASDC)) #Δ_cDC1b
    x[8] = rand(truncated(d.dΔ_DC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/d.R_ASDC)*d.R_ASDC)) #Δ_DC2b
    x[4] = rand(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (x[1] - x[5] -x[6]-x[7]/d.R_ASDC-x[8]/d.R_ASDC)*d.R_ASDC)) #δ_ASDCb
    
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
    
 
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[5]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  b[1] - b[5]) ,b[6]) #Δ_DC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[5] -b[6])*d.R_ASDC), b[7]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_DC2b,0.0, -3e-12 + (b[1] - b[5] -b[6]-b[7]/d.R_ASDC)*d.R_ASDC), b[8]) #Δ_DC2b
    l += logpdf(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (b[1] - b[5] -b[6]-b[7]/d.R_ASDC-b[8]/d.R_ASDC)*d.R_ASDC), b[4]) #δ_ASDCb
  
    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_ASDCbm ,b[1]) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_DC2bm ,b[3]) #p_DC2bm
    
 
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[5]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_DC2bm, 0.0, -5e-12 +  b[1] - b[5]) ,b[6]) #Δ_DC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[5] -b[6])*d.R_ASDC), b[7]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_DC2b,0.0, -3e-12 + (b[1] - b[5] -b[6]-b[7]/d.R_ASDC)*d.R_ASDC), b[8]) #Δ_DC2b
    l += logpdf(truncated(d.dδ_ASDCb,0.0, -2e-12 +  (b[1] - b[5] -b[6]-b[7]/d.R_ASDC-b[8]/d.R_ASDC)*d.R_ASDC), b[4]) #δ_ASDCb
  
    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_ASDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_DC2bm::ContinuousUnivariateDistribution
    dδ_ASDCb::ContinuousUnivariateDistribution
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_DC2bm::ContinuousUnivariateDistribution
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_DC2b::ContinuousUnivariateDistribution
    R_ASDC::Float64
    R_precDC1bm::Float64
    R_preDC2bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_ASDCbm)(x[1]) #p_ASDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_DC2bm)(x[3]) #p_DC2bm
    
    # lower_p_ASDCbm = (λ_ASDC + x[7] + x[8])
    # λ_ASDC = (x[9] + x[10] + x[4]) / d.R_ASDC
    
    y[5] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[5]) #Δ_cDC1bm
    y[6] = bijector(truncated(b.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[5]) )(x[6]) #Δ_DC2bm
    y[7] = bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*b.R_ASDC))(x[7]) #Δ_cDC1b
    y[8] = bijector(truncated(b.dΔ_DC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_ASDC)*b.R_ASDC))(x[8]) #Δ_DC2b
    y[4] = bijector(truncated(b.dδ_ASDCb,0.0,  -2e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_ASDC-x[8]/b.R_ASDC)*b.R_ASDC))(x[4]) #δ_ASDCb

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_ASDCbm))(y[1]) #p_ASDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_DC2bm))(y[3]) #p_DC2bm
    
    x[5] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[5]) #Δ_cDC1bm
    x[6] = inv(bijector(truncated(b.orig.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[5]) ))(y[6]) #Δ_DC2bm
    x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*b.orig.R_ASDC)))(y[7]) #Δ_cDC1b
    x[8] = inv(bijector(truncated(b.orig.dΔ_DC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/b.orig.R_ASDC)*b.orig.R_ASDC)))(y[8]) #Δ_DC2b
    x[4] = inv(bijector(truncated(b.orig.dδ_ASDCb,0.0, -2e-12 +  (x[1] - x[5] -x[6]-x[7]/b.orig.R_ASDC-x[8]/b.orig.R_ASDC)*b.orig.R_ASDC)))(y[4]) #δ_ASDCb
    
    
    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_ASDCbm),x[1]) #p_ASDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_DC2bm),x[3]) #p_DC2bm
    
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])),x[5]) #Δ_cDC1bm
    l += logabsdetjac(bijector(truncated(b.dΔ_DC2bm, 0.0, -5e-12 +  x[1] - x[5])),x[6]) #Δ_DC2bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[5] -x[6])*b.R_ASDC)),x[7]) #Δ_cDC1b
    l += logabsdetjac(bijector(truncated(b.dΔ_DC2b,0.0, -3e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_ASDC)*b.R_ASDC)),x[8]) #Δ_DC2b
    l += logabsdetjac(bijector(truncated(b.dδ_ASDCb,0.0,  -2e-12 + (x[1] - x[5] -x[6]-x[7]/b.R_ASDC-x[8]/b.R_ASDC)*b.R_ASDC)),x[4]) #δ_ASDCb

    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_ASDCbm,d.dp_cDC1bm,d.dp_DC2bm,d.dδ_ASDCb,d.dΔ_cDC1bm,d.dΔ_DC2bm,d.dΔ_cDC1b,d.dΔ_DC2b,d.R_ASDC,d.R_precDC1bm,d.R_preDC2bm)



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
    @unpack R_ASDC, R_cDC1, R_DC2, R_precDC1bm, R_preDC2bm, R_precDC1b, R_preDC2b = metadata.R
    
    ### priors
    prior_dist = MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_DC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),R_ASDC, R_precDC1bm,R_preDC2bm)
    par = Vector{Array{T,1}}(undef, metadata.n_indv)
    for j in 1:metadata.n_indv
        par[j] ~ prior_dist
    end
        
    p_ASDCbm, p_cDC1bm, p_DC2bm, δ_ASDCb, Δ_cDC1bm, Δ_DC2bm, Δ_cDC1b, Δ_DC2b = assign_par(par, length(prior_dist), metadata.n_indv)          
    λ_ASDC = (Δ_cDC1b .+ Δ_DC2b .+ δ_ASDCb) ./ R_ASDC
    
    σ ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf),3, metadata.n_indv)


    ### compound parameter
    δ_ASDCbm = p_ASDCbm .- λ_ASDC .- Δ_cDC1bm .- Δ_DC2bm
    δ_cDC1bm = p_cDC1bm .+ Δ_cDC1bm .* R_precDC1bm
    δ_DC2bm = p_DC2bm .+ Δ_DC2bm .* R_preDC2bm
    δ_cDC1b = Δ_cDC1b .* R_precDC1b
    δ_DC2b = Δ_DC2b .* R_preDC2b

    theta = [[p_ASDCbm[j], δ_ASDCbm[j], p_cDC1bm[j], δ_cDC1bm[j], p_DC2bm[j], δ_DC2bm[j], δ_ASDCb[j], δ_cDC1b[j], δ_DC2b[j], λ_ASDC[j], Δ_cDC1bm[j], Δ_DC2bm[j], Δ_cDC1b[j], Δ_DC2b[j]] for j in 1:metadata.n_indv]

    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## hierachical error
    ### sd infered
    σ_tech ~ filldist(TruncatedNormal(0.0, 1.0, 0.0,Inf), metadata.n_meassurements)
    μ ~ MvNormal(map(j -> sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]], 1:length(metadata.order.donor)), map(j -> σ[metadata.order.population[j], metadata.order.donor[j]], 1:length(metadata.order.donor)))


    ## calculate likelihood
    data ~ MvNormal(μ, σ_tech[metadata.order.technical])

   

    

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf.(Normal.(μ, σ_tech[metadata.order.technical]), data),
    parameters =(;p_ASDCbm=p_ASDCbm, δ_ASDCbm=δ_ASDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_DC2bm=p_DC2bm, δ_DC2bm=δ_DC2bm, δ_ASDCb=δ_ASDCb, δ_cDC1b=δ_cDC1b, δ_DC2b=δ_DC2b, λ_ASDC=λ_ASDC, Δ_cDC1bm=Δ_cDC1bm, Δ_DC2bm=Δ_DC2bm, Δ_cDC1b=Δ_cDC1b, Δ_DC2b=Δ_DC2b))
end


par_range = (;p_ASDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_DC2bm = (0.0,1.0),
δ_ASDCb = (0.0,1.0),
Δ_cDC1bm = (0.0,2.0),
Δ_DC2bm = (0.0,2.0),
Δ_cDC1b = (0.0,2.0),
Δ_DC2b = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
