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
    dΔ_cDC1bm::ContinuousUnivariateDistribution
    dΔ_DC2bm::ContinuousUnivariateDistribution
    dΔ_cDC1b::ContinuousUnivariateDistribution
    dΔ_DC2b::ContinuousUnivariateDistribution
    RASDC::Float64
    RASDC_cDC1_bm::Float64
    RASDC_DC2_bm::Float64
end

function Base.rand(rng::Random.AbstractRNG, d::MyDistribution)
    p_ASDCbm = rand(d.dp_ASDCbm) #p_ASDCbm
    p_cDC1bm = rand(d.dp_cDC1bm) #p_cDC1bm
    p_DC2bm = rand(d.dp_DC2bm) #p_DC2bm
    
    Δ_cDC1bm = rand(truncated(d.dΔ_cDC1bm, 0.0, p_ASDCbm)) #Δ_cDC1bm
    Δ_DC2bm = rand(truncated(d.dΔ_DC2bm, 0.0, p_ASDCbm - Δ_cDC1bm)) #Δ_DC2bm
    Δ_cDC1b = rand(truncated(d.dΔ_cDC1b,0.0, (p_ASDCbm - Δ_cDC1bm -Δ_DC2bm)*d.RASDC)) #Δ_cDC1b
    Δ_DC2b = rand(truncated(d.dΔ_DC2b,0.0, (p_ASDCbm - Δ_cDC1bm -Δ_DC2bm-Δ_cDC1b/d.RASDC)*d.RASDC)) #Δ_DC2b
    δ_ASDCb = rand(truncated(d.dδ_ASDCb,0.0, (p_ASDCbm - Δ_cDC1bm -Δ_DC2bm-Δ_cDC1b/d.RASDC-Δ_DC2b/d.RASDC)*d.RASDC)) #δ_ASDCb
    
    
    upper_λ_cDC1 = p_cDC1bm + Δ_cDC1bm * d.RASDC_cDC1_bm
    upper_λ_DC2 = p_DC2bm + Δ_DC2bm * d.RASDC_DC2_bm
    λ_cDC1 = rand(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1)) #λ_cDC1
    λ_DC2 = rand(truncated(d.dλ_DC2, -Inf, upper_λ_DC2)) #λ_DC2 
    return [p_ASDCbm, p_cDC1bm, p_DC2bm, δ_ASDCb, λ_cDC1, λ_DC2, Δ_cDC1bm, Δ_DC2bm, Δ_cDC1b, Δ_DC2b]
end

function Distributions.logpdf(d::MyDistribution, b::AbstractVector{<:Real})
    p_ASDCbm, p_cDC1bm, p_DC2bm, δ_ASDCb, λ_cDC1, λ_DC2, Δ_cDC1bm, Δ_DC2bm, Δ_cDC1b, Δ_DC2b = b 
    
    l = logpdf(d.dp_ASDCbm ,p_ASDCbm) #p_ASDCbm
    l += logpdf(d.dp_cDC1bm ,p_cDC1bm) #p_cDC1bm
    l += logpdf(d.dp_DC2bm ,p_DC2bm) #p_DC2bm
    
    l += logpdf(truncated(d.dΔ_cDC1bm, 0.0, p_ASDCbm) ,Δ_cDC1bm) #Δ_cDC1bm
    l += logpdf(truncated(d.dΔ_DC2bm, 0.0, p_ASDCbm - Δ_cDC1bm) ,Δ_DC2bm) #Δ_DC2bm
    l += logpdf(truncated(d.dΔ_cDC1b,0.0, (p_ASDCbm - Δ_cDC1bm -Δ_DC2bm)*d.RASDC), Δ_cDC1b) #Δ_cDC1b
    l += logpdf(truncated(d.dΔ_DC2b,0.0, (p_ASDCbm - Δ_cDC1bm -Δ_DC2bm-Δ_cDC1b/d.RASDC)*d.RASDC), Δ_DC2b) #Δ_DC2b
    l += logpdf(truncated(d.dδ_ASDCb,0.0, (p_ASDCbm - Δ_cDC1bm -Δ_DC2bm-Δ_cDC1b/d.RASDC-Δ_DC2b/d.RASDC)*d.RASDC), δ_ASDCb) #δ_ASDCb
    
    
    upper_λ_cDC1 = p_cDC1bm + Δ_cDC1bm * d.RASDC_cDC1_bm
    upper_λ_DC2 = p_DC2bm + Δ_DC2bm * d.RASDC_DC2_bm
    l += logpdf(truncated(d.dλ_cDC1, -Inf, upper_λ_cDC1), λ_cDC1) #λ_cDC1
    l += logpdf(truncated(d.dλ_DC2, -Inf, upper_λ_DC2), λ_DC2) #λ_DC2 

    return l
end

function Bijectors.bijector(d::MyDistribution)
    # define bijector that transforms β based on the value of α
    # mask = Bijectors.PartitionMask(2, [2], [1]) # 2 elements in vector, 2nd entry transformed based on 1st
    mask_δ_ASDCb = Bijectors.PartitionMask(10, [4], [1,7,8,9,10])
    mask_λ_cDC1 = Bijectors.PartitionMask(10, [5], [2,7])
    mask_λ_DC2 = Bijectors.PartitionMask(10, [6], [3,8])
    mask_Δ_cDC1bm = Bijectors.PartitionMask(10, [7], [1])
    mask_Δ_DC2bm = Bijectors.PartitionMask(10, [8], [1,7])
    mask_Δ_cDC1b = Bijectors.PartitionMask(10, [9], [1,7,8])
    mask_Δ_DC2b = Bijectors.PartitionMask(10, [10], [1,7,8,9])
    

    # β_bijector = Bijectors.Coupling(mask) do α
    #     return Bijectors.TruncatedBijector(-α, α)
    # end
    bijector_δ_ASDCb = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(0, x), mask_δ_ASDCb)
    bijector_λ_cDC1 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(-Inf, x[1] + x[2] * d.RASDC_cDC1_bm), mask_λ_cDC1)
    bijector_λ_DC2 = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(-Inf, x[1] + x[2] * d.RASDC_DC2_bm), mask_λ_DC2)
    bijector_Δ_cDC1bm = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(0, x[1]), mask_Δ_cDC1bm)
    bijector_Δ_DC2bm = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(0, x[1]-x[2]), mask_Δ_DC2bm)
    bijector_Δ_cDC1b = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(0, (x[1] - x[2] -x[3])*d.RASDC), mask_Δ_cDC1b)
    bijector_Δ_DC2b = Bijectors.Coupling(x -> Bijectors.TruncatedBijector(0, (x[1] - x[2] -x[3]-x[4]/d.RASDC)*d.RASDC), mask_Δ_DC2b)

    # bijector that transforms the parameter independent from any other parameters and does not transform any other parameter either
    p_ASDCbm_bijector = stack_transform(Bijectors.TruncatedBijector{1}(0.0, Inf), Identity{0}(), Identity{0}(), Identity{0}(), Identity{0}(),Identity{0}(),Identity{0}(), Identity{0}(), Identity{0}(), Identity{0}())
    p_cDC1bm_bijector = stack_transform(Identity{0}(), Bijectors.TruncatedBijector{1}(0.0, Inf), Identity{0}(), Identity{0}(), Identity{0}(),Identity{0}(),Identity{0}(), Identity{0}(), Identity{0}(), Identity{0}())
    p_DC2bm_bijector = stack_transform(Identity{0}(), Identity{0}(), Bijectors.TruncatedBijector{1}(0.0, Inf), Identity{0}(), Identity{0}(),Identity{0}(),Identity{0}(), Identity{0}(), Identity{0}(), Identity{0}())


    return p_ASDCbm_bijector ∘ p_cDC1bm_bijector ∘ p_DC2bm_bijector ∘ bijector_δ_ASDCb ∘ bijector_λ_cDC1 ∘ bijector_λ_DC2 ∘ bijector_Δ_cDC1bm ∘ bijector_Δ_DC2bm ∘ bijector_Δ_cDC1b ∘ bijector_Δ_DC2b
end


function prob_func(prob, theta, label_p, saveat)
    return remake(prob, u0=prob.u0, p=[theta...,label_p...], saveat=saveat, d_discontinuities=[0.5/24.0,label_p[4]])
end

function solve_dc_ode(ODEprob::DiffEqBase.ODEProblem, theta, label_p::Array{Array{Float64,1},1}, timepoints::Array{Array{Float64,1},1}, parallel_mode; solver = AutoVern9(KenCarp4(autodiff=true),lazy=false),kwargs...)
    tmp_prob = EnsembleProblem(ODEprob,prob_func= (prob,i,repeat) -> prob_func(prob, theta[i], label_p[i], timepoints[i]))
    return DifferentialEquations.solve(tmp_prob,solver, parallel_mode; save_idxs=[4,5,6], trajectories=length(label_p), kwargs...)
end



@model function _turing_model(data::Array{Float64,1}, metadata::NamedTuple, ode_prob::ODEProblem, solver, priors::NamedTuple; ode_parallel_mode=EnsembleSerial(), ode_args = (;))
	### priors
    par ~ filldist(MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_DC2bm, Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0), Uniform(0.0,2.0),metadata.R.RASDC, metadata.R.RASDC_cDC1_bm,metadata.R.RASDC_DC2_bm),metadata.n_indv)
    
    p_ASDCbm, p_cDC1bm, p_DC2bm, δ_ASDCb, λ_cDC1, λ_DC2, Δ_cDC1bm, Δ_DC2bm, Δ_cDC1b, Δ_DC2b = eachrow(par)
    λ_ASDC = (Δ_cDC1b .+ Δ_DC2b .+ δ_ASDCb) ./ metadata.R.RASDC
    
    σ ~ filldist(InverseGamma(2,3),3,metadata.n_indv)

    ### compound parameter
    δ_ASDCbm = p_ASDCbm .- λ_ASDC .- Δ_cDC1bm .- Δ_DC2bm
    δ_cDC1bm = p_cDC1bm .+ Δ_cDC1bm .* metadata.R.RASDC_cDC1_bm .- λ_cDC1
    δ_DC2bm = p_DC2bm .+ Δ_DC2bm .* metadata.R.RASDC_DC2_bm .- λ_DC2
    δ_cDC1b = λ_cDC1 .* metadata.R.RcDC1 .+ Δ_cDC1b .* metadata.R.RASDC_cDC1_blood
    δ_DC2b = λ_DC2 .* metadata.R.RDC2 .+ Δ_DC2b .* metadata.R.RASDC_DC2_blood

    ## parameter vector
    theta = [[p_ASDCbm[j], δ_ASDCbm[j], p_cDC1bm[j], δ_cDC1bm[j], p_DC2bm[j], δ_DC2bm[j], δ_ASDCb[j], δ_cDC1b[j], δ_DC2b[j], λ_ASDC[j], λ_cDC1[j], λ_DC2[j], Δ_cDC1bm[j], Δ_DC2bm[j], Δ_cDC1b[j], Δ_DC2b[j]] for j in 1:metadata.n_indv]


    ## solve ODE threaded
    sol = solve_dc_ode(ode_prob, theta, metadata.label_p, metadata.timepoints, ode_parallel_mode, solver=solver; dense=false, ode_args...)

    
    ## exit sample if ODE solver failed
    if any([j.retcode != :Success for j in sol])
        Turing.@addlogprob! -Inf
        return
    end

    ## calculate likelihood (mean)
    data ~ MvNormal([sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]] for j in 1:metadata.n_meassurements], [σ[metadata.order.population[j], metadata.order.donor[j]] for j in 1:metadata.n_meassurements])

   

    ## generated_quantities
    return (;sol =sol,
    log_likelihood = logpdf(MvNormal([sol[metadata.order.donor[j]][metadata.order.population[j], metadata.order.timepoint_idx[j]] for j in 1:metadata.n_meassurements], [σ[metadata.order.population[j], metadata.order.donor[j]] for j in 1:metadata.n_meassurements]), data),
    parameters =(;p_ASDCbm=p_ASDCbm, δ_ASDCbm=δ_ASDCbm, p_cDC1bm=p_cDC1bm, δ_cDC1bm=δ_cDC1bm, p_DC2bm=p_DC2bm, δ_DC2bm=δ_DC2bm, δ_ASDCb=δ_ASDCb, δ_cDC1b=δ_cDC1b, δ_DC2b=δ_DC2b, λ_ASDC=λ_ASDC, λ_cDC1=λ_cDC1, λ_DC2=λ_DC2, Δ_cDC1bm=Δ_cDC1bm, Δ_DC2bm=Δ_DC2bm, Δ_cDC1b=Δ_cDC1b, Δ_DC2b=Δ_DC2b))
end


par_range = (;p_ASDCbm = (0.0,1.0),
p_cDC1bm = (0.0,1.0),
p_DC2bm = (0.0,1.0),
δ_ASDCb = (0.0,1.0),
λ_cDC1 = (0.0,2.0),
λ_DC2 = (0.0,2.0),
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
