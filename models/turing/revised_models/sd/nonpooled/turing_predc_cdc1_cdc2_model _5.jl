using Turing
using DifferentialEquations
using Distributions, Bijectors
using Random

struct MyDistribution <: ContinuousMultivariateDistribution
    dp_preDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_preDCb::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dΔ_cDC2b::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    RpreDC::Float64
    RpreDC_cDC1_bm::Float64
    RpreDC_cDC2_bm::Float64
end

function Distributions.length(d::MyDistribution)
    8# d.n
end

function Base.rand(d::MyDistribution)
	b = zeros(length(d))

    b[1] = rand(d.dp_preDCbm) #p_preDCbm
    b[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    b[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    # b[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1])) #Δ_cDC1bm
    b[7] = rand(truncated(d.dΔ_cDC2bm, 0.0, -6e-12 +  b[1])) #Δ_cDC2bm
    # b[9] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[7] -b[8])*d.RpreDC)) #Δ_cDC1b
    b[8] = rand(truncated(d.dΔ_cDC2b,0.0, -5e-12 + (b[1] - b[7])*d.RpreDC)) #Δ_cDC2b
    b[4] = rand(truncated(d.dδ_preDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8]/d.RpreDC)*d.RpreDC)) #δ_preDCb
    
    
    upper_λ_cDC1 = b[2]
    upper_λ_cDC2 = b[3] + b[7] * d.RpreDC_cDC2_bm
    b[5] = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    b[6] = rand(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2)) #λ_cDC2 
    return b
end

function Distributions._rand!(rng::Random.AbstractRNG,d::MyDistribution, x::Array{Float64,1})
    x[1] = rand(d.dp_preDCbm) #p_preDCbm
    x[2] = rand(d.dp_cDC1bm) #p_cDC1bm
    x[3] = rand(d.dp_cDC2bm) #p_cDC2bm
    
    # x[7] = rand(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])) #Δ_cDC1bm
    x[7] = rand(truncated(d.dΔ_cDC2bm, 0.0, -6e-12 +  x[1])) #Δ_cDC2bm
    # x[9] = rand(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*d.RpreDC)) #Δ_cDC1b
    x[8] = rand(truncated(d.dΔ_cDC2b,0.0, -5e-12 + (x[1] - x[7])*d.RpreDC)) #Δ_cDC2b
    x[4] = rand(truncated(d.dδ_preDCb,0.0, -4e-12 +  (x[1] - x[7] -x[8]/d.RpreDC)*d.RpreDC)) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3] + x[7] * d.RpreDC_cDC2_bm
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
    l = logpdf(d.dp_preDCbm ,b[1]) #p_preDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
    # l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC2bm
    # l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[7] -b[8])*d.RpreDC), b[9]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_cDC2b,0.0, -5e-12 + (b[1] - b[7])*d.RpreDC), b[8]) #Δ_cDC2b
    l += logpdf(truncated(d.dδ_preDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8]/d.RpreDC)*d.RpreDC), b[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = b[2]
    upper_λ_cDC2 = b[3] + b[7] * d.RpreDC_cDC2_bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector)
    l = logpdf(d.dp_preDCbm ,b[1]) #p_preDCbm
    l += logpdf(d.dp_cDC1bm ,b[2]) #p_cDC1bm
    l += logpdf(d.dp_cDC2bm ,b[3]) #p_cDC2bm
    
    # l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_cDC2bm, 0.0, -6e-12 +  b[1]) ,b[7]) #Δ_cDC2bm
    # l += logpdf(truncated(d.dΔ_cDC1b,0.0, -4e-12 + (b[1] - b[7] -b[8])*d.RpreDC), b[9]) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_cDC2b,0.0, -5e-12 + (b[1] - b[7])*d.RpreDC), b[8]) #Δ_cDC2b
    l += logpdf(truncated(d.dδ_preDCb,0.0, -4e-12 +  (b[1] - b[7] -b[8]/d.RpreDC)*d.RpreDC), b[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = b[2]
    upper_λ_cDC2 = b[3] + b[7] * d.RpreDC_cDC2_bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), b[5]) #λ_cDC1
    l += logpdf(truncated(d.dλ_cDC2, -Inf, upper_λ_cDC2), b[6]) #λ_cDC2 

    return l
end

struct MyBijector <: Bijectors.Bijector{1} 
    dp_preDCbm::ContinuousUnivariateDistribution
    dp_cDC1bm::ContinuousUnivariateDistribution
    dp_cDC2bm::ContinuousUnivariateDistribution
    dδ_preDCb::ContinuousUnivariateDistribution
    dΔ_cDC2bm::ContinuousUnivariateDistribution
    dΔ_cDC2b::ContinuousUnivariateDistribution
    dλ_cDC1::ContinuousUnivariateDistribution
    dλ_cDC2::ContinuousUnivariateDistribution
    RpreDC::Float64
    RpreDC_cDC1_bm::Float64
    RpreDC_cDC2_bm::Float64
end

function (b::MyBijector)(x::AbstractVector)
	y = similar(x)

    y[1] = bijector(b.dp_preDCbm)(x[1]) #p_preDCbm
    y[2] = bijector(b.dp_cDC1bm)(x[2]) #p_cDC1bm
    y[3] = bijector(b.dp_cDC2bm)(x[3]) #p_cDC2bm
    
    # y[7] = bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 + x[1]) )(x[7]) #Δ_cDC1bm
    y[7] = bijector(truncated(b.dΔ_cDC2bm, 0.0, -6e-12 +  x[1]) )(x[7]) #Δ_cDC2bm
    # y[9] = bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.RpreDC))(x[9]) #Δ_cDC1b
    y[8] = bijector(truncated(b.dΔ_cDC2b,0.0, -5e-12 + (x[1] - x[7])*b.RpreDC))(x[8]) #Δ_cDC2b
    y[4] = bijector(truncated(b.dδ_preDCb,0.0,  -4e-12 + (x[1] - x[7] -x[8]/b.RpreDC)*b.RpreDC))(x[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3] + x[7] * b.RpreDC_cDC2_bm
    y[5] = bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1))(x[5]) #λ_cDC1
    y[6] = bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2))(x[6]) #λ_cDC2 

    return y
end
function (b::Inverse{<:MyBijector})(y::AbstractVector)
	x = similar(y)

    x[1] = inv(bijector(b.orig.dp_preDCbm))(y[1]) #p_preDCbm
    x[2] = inv(bijector(b.orig.dp_cDC1bm))(y[2]) #p_cDC1bm
    x[3] = inv(bijector(b.orig.dp_cDC2bm))(y[3]) #p_cDC2bm
    
    # x[7] = inv(bijector(truncated(b.orig.dΔ_cDC1bm, 0.0,  -6e-12 + x[1]) ))(y[7]) #Δ_cDC1bm
    x[7] = inv(bijector(truncated(b.orig.dΔ_cDC2bm, 0.0, -6e-12 +  x[1]) ))(y[7]) #Δ_cDC2bm
    # x[9] = inv(bijector(truncated(b.orig.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.orig.RpreDC)))(y[9]) #Δ_cDC1b
    x[8] = inv(bijector(truncated(b.orig.dΔ_cDC2b,0.0, -5e-12 + (x[1] - x[7])*b.orig.RpreDC)))(y[8]) #Δ_cDC2b
    x[4] = inv(bijector(truncated(b.orig.dδ_preDCb,0.0, -4e-12 +  (x[1] - x[7] -x[8]/b.orig.RpreDC)*b.orig.RpreDC)))(y[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3] + x[7] * b.orig.RpreDC_cDC2_bm
    x[5] = inv(bijector(truncated(b.orig.dλ_cDC1, -Inf, upper_λ_cDC1)))(y[5]) #λ_cDC1
    x[6] = inv(bijector(truncated(b.orig.dλ_cDC2, -Inf, upper_λ_cDC2)))(y[6]) #λ_cDC2 

    return x
end
function Bijectors.logabsdetjac(b::MyBijector, x::AbstractVector)
	l = float(zero(eltype(x)))

    l += logabsdetjac(bijector(b.dp_preDCbm),x[1]) #p_preDCbm
    l += logabsdetjac(bijector(b.dp_cDC1bm),x[2]) #p_cDC1bm
    l += logabsdetjac(bijector(b.dp_cDC2bm),x[3]) #p_cDC2bm
    
    # l += logabsdetjac(bijector(truncated(b.dΔ_cDC1bm, 0.0, -6e-12 +  x[1])),x[7]) #Δ_cDC1bm
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2bm, 0.0, -6e-12 +  x[1])),x[7]) #Δ_cDC2bm
    # l += logabsdetjac(bijector(truncated(b.dΔ_cDC1b,0.0, -4e-12 + (x[1] - x[7] -x[8])*b.RpreDC)),x[9]) #Δ_cDC1b
    l += logabsdetjac(bijector(truncated(b.dΔ_cDC2b,0.0, -5e-12 + (x[1] - x[7])*b.RpreDC)),x[8]) #Δ_cDC2b
    l += logabsdetjac(bijector(truncated(b.dδ_preDCb,0.0,  -4e-12 + (x[1] - x[7] -x[8]/b.RpreDC)*b.RpreDC)),x[4]) #δ_preDCb
    
    
    upper_λ_cDC1 = x[2]
    upper_λ_cDC2 = x[3] + x[7] * b.RpreDC_cDC2_bm
    l += logabsdetjac(bijector(truncated(b.dλ_cDC1, -Inf, upper_λ_cDC1)),x[5]) #λ_cDC1
    l += logabsdetjac(bijector(truncated(b.dλ_cDC2, -Inf, upper_λ_cDC2)),x[6]) #λ_cDC2 


    return l
end
Bijectors.bijector(d::MyDistribution)= MyBijector(d.dp_preDCbm,d.dp_cDC1bm,d.dp_cDC2bm,d.dδ_preDCb,d.dΔ_cDC2bm,d.dΔ_cDC2b,d.dλ_cDC1,d.dλ_cDC2,d.RpreDC,d.RpreDC_cDC1_bm,d.RpreDC_cDC2_bm)



function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat, save_idxs=save_idxs,d_discontinuity=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode; solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=[4,5,6], trajectories=length(label_p), kwargs...)
end



@model function _turing_model(data::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple, sim::Bool=false, logp::Bool=false; ode_args = (;))
	### priors
    par ~ MyDistribution(priors.p_preDCbm, priors.p_cDC1bm, priors.p_cDC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),data.R.RpreDC, data.R.RpreDC_cDC1_bm,data.R.RpreDC_cDC2_bm)
    p_preDCbm, p_cDC1bm, p_cDC2bm, δ_preDCb, λ_cDC1, λ_cDC2, Δ_cDC2bm, Δ_cDC2b = eachrow(par)           
    λ_preDC = (Δ_cDC2b .+ δ_preDCb) ./ data.R.RpreDC
    
    σ1 ~ TruncatedNormal(0.0, 1.0, 0.0,Inf)
    σ2 ~ TruncatedNormal(0.0, 1.0, 0.0,Inf)
    σ3 ~ TruncatedNormal(0.0, 1.0, 0.0,Inf)


    ### compound parameter
    δ_preDCbm = p_preDCbm .- λ_preDC .-  Δ_cDC2bm
    δ_cDC1bm = p_cDC1bm .- λ_cDC1
    δ_cDC2bm = p_cDC2bm .+ Δ_cDC2bm .* data.R.RpreDC_cDC2_bm .- λ_cDC2
    δ_cDC1b = λ_cDC1 .* data.R.RcDC1
    δ_cDC2b = λ_cDC2 .* data.R.RcDC2 .+ Δ_cDC2b .* data.R.RpreDC_cDC2_blood

    theta = [[p_preDCbm[j], δ_preDCbm[j], p_cDC1bm[j], δ_cDC1bm[j], p_cDC2bm[j], δ_cDC2bm[j], δ_preDCb[j], δ_cDC1b[j], δ_cDC2b[j], λ_preDC[j], λ_cDC1[j], λ_cDC2[j], Δ_cDC2bm[j], Δ_cDC2b[j]] for j in 1:metadata.n_indv]
    
    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)


    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## hierachical error
    ### sd infered
    σ_tech ~ filldist(InverseGamma(2, 3), metadata.n_meassurements)
    μ ~ MvNormal(sol[metadata.order.donor][metadata.order.population, metadata.order.timepoint_idx], σ[metadata.order.population, metadata.order.donor])


    ## calculate likelihood
    data ~ MvNormal(μ, σ_tech[metadata.order.technical])

   
   

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf(MvNormal(μ, σ_tech[metadata.order.technical]), data),
    parameters =(;p_preDCbm=p_preDCbm, δ_preDCbm=δ_preDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_cDC2bm=p_cDC2bm, δ_cDC2bm=δ_cDC2bm, δ_preDCb=δ_preDCb, δ_cDC1b=δ_cDC1b, δ_cDC2b=δ_cDC2b, λ_preDC=λ_preDC, λ_cDC1=λ_cDC1, λ_cDC2=λ_cDC2, Δ_cDC1bm=Δ_cDC1bm, Δ_cDC2bm=Δ_cDC2bm, Δ_cDC1b=Δ_cDC1b, Δ_cDC2b=Δ_cDC2b))
end


par_range = (;p_preDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_cDC2bm = (0.0,1.0),
δ_preDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_cDC2 = (0.0,2.0),
Δ_cDC2bm = (0.0,2.0),
Δ_cDC2b = (0.0,2.0),
σ1 = (0.0,2.0),
σ2 = (0.0,2.0),
σ3 = (0.0,2.0))


par_range_names = keys(par_range)
par_lb = [par_range[j][1] for j in par_range_names]
par_ub = [par_range[j][2] for j in par_range_names]
