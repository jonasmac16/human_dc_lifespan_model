### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ c63baab2-716f-11eb-2aa2-295e1380384b
using DrWatson

# ╔═╡ e0401a03-70b0-415f-9223-202c1f4aef4f
begin
	DrWatson.@quickactivate "Model of DC Differentiation"
	using DifferentialEquations
	using ModelingToolkit
	using Turing
	using DataFrames
	using DataFramesMeta
	using Distributions
	using Plots
	using StatsPlots
	using Images
	using CSV
	using JLSO
	using MCMCChains
	# using RCall
	using DelimitedFiles
	using Pipe: @pipe
	using BenchmarkTools
	using Logging
	include(srcdir("mcmcchains_diagnostics.jl"))
end

# ╔═╡ e4e12340-7172-11eb-22b0-f7a72dfbb863
include(projectdir("models", "ode", "U_func_c.jl"))

# ╔═╡ 0b6b2120-7171-11eb-2e58-ff6d10cf5254
begin
	include(srcdir("dataprep.jl"))
	include(srcdir("create_dist.jl"))
end

# ╔═╡ 08c11aec-77a7-11eb-361a-618885fecfcb
donor_ids = ["D01", "D02", "D04", "C66", "C67", "C68", "C55"]

# ╔═╡ 8b6fd220-7aee-11eb-3954-99cf3a51bc7b
cell_cycle_approach = 3

# ╔═╡ 83bcdb10-7c1a-11eb-3b90-9549e1fd68f9
ratio_approach = "1c"

# ╔═╡ 8799db66-7c1a-11eb-22b4-417bc8a070a5
ratio_summary = "median"

# ╔═╡ 99e32cfa-7c1a-11eb-1392-0143e6127ac4
model_id = "4"

# ╔═╡ caa78152-716f-11eb-2937-2f55801a9180
md"""
## Purpose
Fitting the new implementation of model $(model_id) and the corresponding *Turing.jl* model with upper and lower bounds on the proliferation rates for all populations on the basis of cell cycle phase data approach 1.
"""

# ╔═╡ da7b2e6e-784d-11eb-01fa-5f95a00dca93
# load(projectdir("plots", "model_figures","model_"*model_id*".pdf"))

# ╔═╡ ca40c6b4-7170-11eb-3088-eb5f0cff44ea
include(projectdir("models", "ode","revised_models", "model_"*model_id*".jl"))

# ╔═╡ f3c8a1a0-7170-11eb-2af4-3b0a7b4989f8
include(projectdir("models", "turing", "revised_models", "mean_student_t", "pooled", "turing_asdc_cdc1_dc2_model_"*model_id*".jl"))

# ╔═╡ f891cf0c-7b40-11eb-0c5f-930711de036e
begin	
	include(projectdir("scripts", "run_project", "00_mcmc_settings.jl"))
	warm_up = Int(mcmc_iters/2)
	sample_iters = mcmc_iters
	accept_rate = 0.98
end

# ╔═╡ 83ac0efc-7ce7-11eb-32bc-c92fa3d52078
begin
	
	parallel_sampling_method = MCMCThreads() #nothing or MCMCThreads()
	solver_parallel_methods = EnsembleThreads() #EnsembleSerial() or #EnsembleThreads() for parallel solving of ODEs
end

# ╔═╡ 1963943e-84e4-11eb-0b7a-bba5cdb5467b
begin
	tau_stop = 3.5/24.0
	bc = 0.73
end

# ╔═╡ 331d6eb6-77a8-11eb-2fc3-893e49435a73
begin
	notebook_folder_title = basename(@__DIR__)
	notebook_folder = joinpath(basename(@__DIR__), "results")
	mkpath(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder))
	data_folder = "data_derek_20210823"
end

# ╔═╡ 9be93270-716f-11eb-1eca-5df3697044c0
md"# $(notebook_folder_title)"

# ╔═╡ 5159f562-7171-11eb-0d12-37cacdc3e84f
begin
	for j in 1:2
		global label_ps = DataFrame(load(datadir("exp_pro", "labeling_parameters_revision.csv")))
	end
end

# ╔═╡ c675c8ec-610c-4333-aee4-31efa9b1cb30
label_ps

# ╔═╡ fb05c902-7c76-11eb-1d23-ed6b66f9ab21
begin
	df_p_priors = CSV.read(datadir("exp_pro", "p_priors_revision.csv"), DataFrame)

	priors = (
		p_ASDCbm = (@pipe df_p_priors |> subset(_, :parameter => (x -> x .== "ASDC")) |> _[1,:] |> create_dist(_.dist, _.μ, _.σ, 1, 8e-12, _.upper)),
		p_cDC1bm = (@pipe df_p_priors |> subset(_, :parameter => (x -> x .== "cDC1")) |> _[1,:] |> create_dist(_.dist, _.μ, _.σ, 1, 8e-12, _.upper)),
		p_DC2bm = (@pipe df_p_priors |> subset(_, :parameter => (x -> x .== "DC2")) |> _[1,:] |> create_dist(_.dist, _.μ, _.σ, 1, 8e-12, _.upper)))
end

# ╔═╡ a3d2f836-7200-11eb-0d49-a7d5a55ab8e5
cell_ratios = DataFrame(load(datadir("exp_pro", "cell_ratios_revision.csv")))

# ╔═╡ a38b9a86-7200-11eb-2dcb-e1b7967f37f0
# cell_cycle = DataFrame(load(datadir("exp_pro","JM_0014", "cell_cycle_status_proliferation_rate_bm.csv")))

# ╔═╡ a34e9244-7200-11eb-0d20-0f8f74e2b6f6
labelling_data = DataFrame(load(datadir("exp_pro", "labelling_data_revision.csv")))

# ╔═╡ 6b6ff7d0-7a7c-11eb-3ac1-fd7a3e6086ac
function aic(logp, npar; k=2)
	aic = k*npar - 2*logp
	return aic
end

# ╔═╡ 40224496-7a7e-11eb-00ca-ab3f5cc47243
function aicc(logp, npar, nsamp; k=2)
	aicc = aic(logp, npar; k=k) + ((2*npar^2+2*npar)/(nsamp-npar-1))
	return aicc
end

# ╔═╡ 21584bf6-7181-11eb-04a9-5742b91cb102
U_func(t, fr, delta, frac, tau) = U_smooth_2stp(t, fr, delta, 0.5/24.0, tau, frac; c = bc)

# ╔═╡ a1e72c1e-7175-11eb-3eee-25fae4864542
u0 = zeros(6)

# ╔═╡ 79d513e6-7176-11eb-0080-f7d16330d1c4
p_init = ones([20,18,16,18,18][tryparse(Int, model_id)])

# ╔═╡ 9c6a5860-717a-11eb-11b4-11109f68f8c9
solver_in = AutoTsit5(KenCarp4())

# ╔═╡ 398c6598-7ccd-11eb-20d2-fbabb2a08251
begin
	Turing.setadbackend(:forwarddiff)

    data_in = prepare_data_turing(labelling_data, cell_ratios, label_ps, tau_stop; population = ["ASDC", "cDC1", "DC2"], individual = donor_ids, label_p_names = [:fr,:delta, :frac], ratio_approach=ratio_approach, ratio_summary = ratio_summary, mean_data = true)
	
	model(du,u,p,t) = eval(Symbol("_model_"*model_id))(du,u,p,t, U_func, data_in.metadata.R)

	problem = ODEProblem(model, u0, (0.0, maximum(vcat(data_in.metadata.timepoints...))),p_init)

	## MTK
	mtk_model = modelingtoolkitize(problem)
	f_opt = ODEFunction(mtk_model, jac=true)
	mtk_problem = ODEProblem(f_opt, u0,(0.0, maximum(vcat(data_in.metadata.timepoints...))),p_init);

	turing_model = _turing_model(data_in.data, data_in.metadata, mtk_problem, solver_in, priors, ode_parallel_mode=solver_parallel_methods; ode_args=(abstol=1e-10, reltol=1e-10, maxiters=1e8))
	
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"mcmc_res.jlso")))
		if isnothing(parallel_sampling_method)
			chains = mapreduce(c -> sample(turing_model, NUTS(warm_up, accept_rate), mcmc_iters, progress=false), chainscat, 1:n_chains)
		else
			chains = sample(turing_model, NUTS(warm_up, accept_rate), parallel_sampling_method, mcmc_iters, n_chains, progress=false);
		end
		JLSO.save(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"mcmc_res.jlso"), :chain => chains)
	else
		chains = JLSO.load(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"mcmc_res.jlso"))[:chain]
	end
end

# ╔═╡ 08cc7a0e-7a86-11eb-138b-c5bcd9a4bd37
md"## Chain diagnostics"

# ╔═╡ c6545e24-85a4-11eb-2262-c3a7ed2a61fb
begin
	io = open(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"check_diagnostics.txt"), "w+")
	logger = SimpleLogger(io)

	with_logger(logger) do
		check_diagnostics(chains[:,:,:])
	end

	flush(io)
	close(io)

	io_read = open(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"check_diagnostics.txt"), "r")
	check_out = readlines(io_read)
	close(io_read)
	
	check_out
end

# ╔═╡ 53543fac-7cd0-11eb-3b38-919570f85aae
begin
	p_diag_1 = plot(chains, title=permutedims(vcat([[j, j] for j in par_range_names]...)), label=permutedims([("Chain " .* string.(collect(1:n_chains)))...]))
	for k in 1:(length(p_init)-10)
		density!(p_diag_1, [rand(MyDistribution(priors.p_ASDCbm, priors.p_cDC1bm, priors.p_DC2bm, [Uniform(0.0,2.0) for j in 1:(length(p_init)-13)]..., data_in.metadata.R.R_ASDC, data_in.metadata.R.R_ASDCcDC1bm,data_in.metadata.R.R_ASDCDC2bm))[k] for j in 1:1000], subplot=(k-1)*2+2, c=:black, legend=true, label="prior")
	end
	savefig(p_diag_1, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"diagnostic_all.pdf"))
	p_diag_1
end

# ╔═╡ a4c3be9f-1561-4d5e-88fb-e36979f27f93
begin
	function create_model_prediction_df(vec_sol)
	df_wide = vcat([(@pipe vec_sol[k].u |> _[j] |> DataFrame(_) |> rename(_, "x₁(t)" => "ASDC_bm", "x₂(t)" => "cDC1_bm", "x₃(t)" => "DC2_bm","x₄(t)"=>"ASDC_b",	"x₅(t)"=>"cDC1_b",	"x₆(t)"=>"DC2_b") |> insertcols!(_, :donor => donor_ids[j], :sample_idx=>k)) for k in 1:length(vec_sol) for j in 1:length(vec_sol[k])]...)

	df_combined = @pipe df_wide |> DataFrames.stack(_, Not([:timestamp, :donor, :sample_idx])) |> transform(_, :variable => ByRow(x -> (;zip((:population, :location),Tuple(split(x, "_")))...))=> AsTable) |> select(_, Not(:variable))
	
	
	return df_combined
	end
end


# ╔═╡ 5b3c6540-77a9-11eb-0c38-376abea08f6b
md"## PPC of all individuals fitted"

# ╔═╡ df988bf9-d8d2-4777-9bad-d75745609c4a
function get_loglikelihood(model, chain)
	[j.log_likelihood for j in generated_quantities(model, chain)]
end

# ╔═╡ 3cd89616-ff22-4a02-961b-3060189402e2
function get_parameters(model, chain)
	[j.parameters for j in generated_quantities(model, chain)]
end

# ╔═╡ 42f913a5-02b4-4c4b-a366-9df35cf8db54
function get_posterior_predictive(model, chain)
	[j.sol for j in generated_quantities(model, chain)]
end

# ╔═╡ 751a2de4-aa17-425b-b23e-0842aee30c44
begin
	function plot_ppc(sols, vars;cols=[:grey, :grey, :grey], subplotkwargs=(;), kwargs...)
		n_indv = length(first(sols))
		
		p = plot(layout=(n_indv,length(vars)); kwargs...)
		
		
		for j in sols
			for (idx, k) in enumerate(j)
				for (idx2, l) in enumerate(vars)
					plot!(p, k, vars=l,colour= cols[idx2], subplot=(idx-1)*3 + idx2; subplotkwargs...) 
				end
			end
		end
		return p
	end
	function plot_ppc_condensed(sols, vars; pop=["preDC", "cDC1", "cDC2"], cols=[:grey, :grey, :grey], subplotkwargs=(;), kwargs...)
		n_indv = length(first(sols))
		
		p = plot(layout=(n_indv,1); kwargs...)
		
		if length(vars) != n_indv
			vars = [vars[1] for j in 1:n_indv]
		end

		for (n_samp, j) in enumerate(sols)
			for (idx, k) in enumerate(j)
				for l in vars[idx]
					plot!(p, k, vars=l,colour= cols[l], subplot=idx, lab= n_samp < length(sols) ? "" : pop[l]*"(pred)" ; subplotkwargs...) 
				end
			end
		end
		return p
	end

end

# ╔═╡ ff300623-cc98-4d79-adac-766d60d8b4da
begin
	function make_labels(pop, donor, pop_label=["preDC", "cDC1", "cDC2"])
		pop=deepcopy(pop)
		donor=deepcopy(donor)
		label_res = fill("", length(pop))

		for j in unique(donor)
			pop_tmp = deepcopy(pop)
			pop_tmp[donor .!= j] .= 0
			for (idx, k) in enumerate(pop_label)
				if !isnothing(findfirst(x-> x == idx, pop_tmp))
					label_res[findfirst(x-> x == idx, pop_tmp)] = k
				end
			end
		end
		return label_res
	end

	function plot_ppc(sols, data, vars; cols=[colorant"#755494", colorant"#de3458", colorant"#4e65a3"],  subplotkwargs=(;), kwargs...)
		p = plot_ppc(sols, vars, cols=cols, subplotkwargs=subplotkwargs; kwargs...)
		# col =  ["red", "blue", "green"]
		
		for j in 1:length(data.data)
			scatter!(p, [data.metadata.timepoints[data.metadata.order.donor[j]][data.metadata.order.timepoint_idx[j]]], [data.data[j]], subplot=(data.metadata.order.donor[j]-1)*3 + data.metadata.order.population[j], c = cols[data.metadata.order.population[j]], yerror=data.data_sd[j])
		end
		return p
	end

	function plot_ppc_condensed(sols, data, vars;pop=["preDC", "cDC1", "cDC2"], cols=[colorant"#755494", colorant"#de3458", colorant"#4e65a3"], subplotkwargs=(;),datakwargs=(;), kwargs...)
		p = plot_ppc_condensed(sols, vars, cols=cols, subplotkwargs=subplotkwargs; kwargs...)
		
		for j in 1:length(data.data)
			scatter!(p,
			[data.metadata.timepoints[data.metadata.order.donor[j]][data.metadata.order.timepoint_idx[j]]], 
			[data.data[j]],
			subplot=data.metadata.order.donor[j], 
			c = cols[data.metadata.order.population[j]], 
			yerror=data.data_sd[j],
			lab=make_labels(data.metadata.order.population, data.metadata.order.donor)[j];
			datakwargs...)
		end
		return p
	end
end

# ╔═╡ e935825e-4671-4a7c-b0d0-c1deb6e036ff
function sample_mcmc(chn, n; replace=true, ordered=false)
	MCMCChains.subset(chn, sample(range(resetrange(chn)), n; replace = replace, ordered = ordered))
end

# ╔═╡ e64f5bf3-acd3-451c-9b04-bc5a80f08a28
begin
	data_ppc = deepcopy(data_in)
	for j in 1:length(data_ppc.metadata.timepoints)
		data_ppc.metadata.timepoints[j] = collect(0.0:0.1:24.0)
	end
	
	turing_model_ppc = _turing_model(data_ppc.data, data_ppc.metadata, mtk_problem, solver_in, priors, ode_parallel_mode=solver_parallel_methods; ode_args=(abstol=1e-10, reltol=1e-10, maxiters=1e8, save_idxs=[1,2,3,4,5,6]))

end	

# ╔═╡ cd1a937c-7951-4bdc-b48a-da490fcbc25d
begin
	if !isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_ppc.csv"))
		ppc = @pipe get_posterior_predictive(turing_model_ppc, sample_mcmc(chains, 50)) |> [_[j] for j in 1:length(_)]
		df_ppc = create_model_prediction_df(ppc)
		save(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_ppc.csv"), df_ppc)
	end
end

# ╔═╡ 29b023e3-879d-438e-bef8-f290bf3d4a55
begin
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit_bm.pdf")))

		# ppc_bm = @pipe get_posterior_predictive(turing_model_ppc_bm, sample_mcmc(chains, 50)) |>[_[j] for j in 1:length(_)]

	p_ppc_bm = plot_ppc(ppc, 1:3, subplotkwargs=(; alpha=0.1);title= permutedims([((permutedims(donor_ids) .* " ") .* ["ASDC (bm)", "cDC1 (bm)","DC2 (bm)"])...]), size=(1000,1000), legend=false)
	else
		load(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit_bm.pdf"))
	end
	
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit.pdf")))

		# ppc_b = @pipe get_posterior_predictive(turing_model, sample_mcmc(chains, 50)) |>[_[j] for j in 1:length(_)]
		p_ppc = plot_ppc(ppc, data_in, 4:6, subplotkwargs=(; alpha=0.1);title= permutedims([((permutedims(donor_ids) .* " ") .* ["ASDC", "cDC1","DC2"])...]), size=(1000,1000), legend=false)
	else
		load(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit.pdf"))
	end
end

# ╔═╡ 46158577-9672-44f8-bcf3-57eec228d5b0
begin
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit.pdf")))
		savefig(p_ppc, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit.pdf"))
		savefig(p_ppc, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit.png"))
		savefig(p_ppc, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit.svg"))
	end
	
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit_bm.pdf")))
		savefig(p_ppc_bm, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit_bm.pdf"))
		savefig(p_ppc_bm, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit_bm.png"))
		savefig(p_ppc_bm, projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"ppc_fit_bm.svg"))
	end
end

# ╔═╡ 64808ea4-746a-45ec-8f3f-a0625a1816db
md"## Parameter dataframe"

# ╔═╡ a64bb46f-189d-466b-83b2-0f17602f5a46
if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_mcmc_comp.jlso")))
	parameter_est = get_parameters(turing_model, chains)
end

# ╔═╡ 69182965-21a3-442a-971e-2e27840a658e
begin
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_mcmc_comp.jlso")))
		df_par_all = DataFrame(p_ASDCbm=Float64[], δ_ASDCbm=Float64[], p_cDC1bm=Float64[], δ_cDC1bm=Float64[], p_DC2bm=Float64[], δ_DC2bm=Float64[], δ_ASDCb=Float64[], δ_cDC1b=Float64[], δ_DC2b=Float64[], λ_ASDC=Float64[], λ_cDC1=Float64[], λ_DC2=Float64[], Δ_cDC1bm=Float64[], Δ_DC2bm=Float64[], Δ_cDC1b=Float64[], Δ_DC2b=Float64[])

		@pipe parameter_est |>
		for j in _
			push!(df_par_all, j, cols=:subset)
		end
		JLSO.save(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_mcmc_comp.jlso"), :df_par_all => df_par_all)
		save(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_mcmc_comp.csv"), df_par_all)
	else
		df_par_all = JLSO.load(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"df_mcmc_comp.jlso"))[:df_par_all]
	
	end
end

# ╔═╡ f1611d14-74d9-4372-b203-53973cef93bf
df_par_all

# ╔═╡ eafc5a84-7da1-11eb-03d9-29a3cac35040
md"## Pointwise loglikehood"

# ╔═╡ 193aee62-7da2-11eb-2ff6-d5562c2a1331
begin

end

# ╔═╡ 82cfa81f-a990-4c55-b05b-f35c7e3649b0
if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"logp_3d_mat.jlso")))
	loglike= get_loglikelihood(turing_model, chains)
end

# ╔═╡ 19b5416e-85b4-11eb-04ea-3551dd6da8a8
begin
	if !(isfile(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"logp_3d_mat.jlso")))

		loglike_2d = @pipe loglike |>
		hcat(_...) |>
		permutedims(_) |>
		Array

		loglike_3d = zeros(size(loglike,1),size(loglike,2), size(loglike_2d,2))

		for j in 1:size(loglike,2)
			loglike_3d[:,j,:] = @pipe loglike |> _[:,1] |> hcat(_...)'
		end

		loglike_3d = permutedims(loglike_3d, [3,1,2])

		for j in 1:size(loglike_3d,3)
			DelimitedFiles.writedlm(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"logp_3d_mat_$(j).txt"), loglike_3d[:,:,j])
		end
		
		JLSO.save(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"logp_3d_mat.jlso"), :loglike_3d=>loglike_3d)
	else
		loglike_3d = JLSO.load(projectdir("notebooks","02_fitting","01_lognormal_prior",notebook_folder,"logp_3d_mat.jlso"))[:loglike_3d]
	end
	loglike_3d
end

# ╔═╡ 2582b3de-58cf-4949-a1a0-f9d45d9b28ee
begin

end

# ╔═╡ 6ac22f4e-85b5-11eb-0674-211b09bb5e57
begin

end

# ╔═╡ bb90f310-716f-11eb-30e3-532345b94219
md"## Libraries"

# ╔═╡ Cell order:
# ╟─9be93270-716f-11eb-1eca-5df3697044c0
# ╠═e0401a03-70b0-415f-9223-202c1f4aef4f
# ╟─caa78152-716f-11eb-2937-2f55801a9180
# ╠═da7b2e6e-784d-11eb-01fa-5f95a00dca93
# ╠═08c11aec-77a7-11eb-361a-618885fecfcb
# ╠═8b6fd220-7aee-11eb-3954-99cf3a51bc7b
# ╠═83bcdb10-7c1a-11eb-3b90-9549e1fd68f9
# ╠═8799db66-7c1a-11eb-22b4-417bc8a070a5
# ╠═99e32cfa-7c1a-11eb-1392-0143e6127ac4
# ╠═f891cf0c-7b40-11eb-0c5f-930711de036e
# ╠═83ac0efc-7ce7-11eb-32bc-c92fa3d52078
# ╠═1963943e-84e4-11eb-0b7a-bba5cdb5467b
# ╠═331d6eb6-77a8-11eb-2fc3-893e49435a73
# ╠═ca40c6b4-7170-11eb-3088-eb5f0cff44ea
# ╠═f3c8a1a0-7170-11eb-2af4-3b0a7b4989f8
# ╠═e4e12340-7172-11eb-22b0-f7a72dfbb863
# ╠═5159f562-7171-11eb-0d12-37cacdc3e84f
# ╠═c675c8ec-610c-4333-aee4-31efa9b1cb30
# ╠═fb05c902-7c76-11eb-1d23-ed6b66f9ab21
# ╠═a3d2f836-7200-11eb-0d49-a7d5a55ab8e5
# ╠═a38b9a86-7200-11eb-2dcb-e1b7967f37f0
# ╠═a34e9244-7200-11eb-0d20-0f8f74e2b6f6
# ╠═0b6b2120-7171-11eb-2e58-ff6d10cf5254
# ╠═6b6ff7d0-7a7c-11eb-3ac1-fd7a3e6086ac
# ╠═40224496-7a7e-11eb-00ca-ab3f5cc47243
# ╠═21584bf6-7181-11eb-04a9-5742b91cb102
# ╠═a1e72c1e-7175-11eb-3eee-25fae4864542
# ╠═79d513e6-7176-11eb-0080-f7d16330d1c4
# ╠═9c6a5860-717a-11eb-11b4-11109f68f8c9
# ╠═398c6598-7ccd-11eb-20d2-fbabb2a08251
# ╟─08cc7a0e-7a86-11eb-138b-c5bcd9a4bd37
# ╠═c6545e24-85a4-11eb-2262-c3a7ed2a61fb
# ╠═53543fac-7cd0-11eb-3b38-919570f85aae
# ╠═a4c3be9f-1561-4d5e-88fb-e36979f27f93
# ╟─5b3c6540-77a9-11eb-0c38-376abea08f6b
# ╠═df988bf9-d8d2-4777-9bad-d75745609c4a
# ╠═3cd89616-ff22-4a02-961b-3060189402e2
# ╠═42f913a5-02b4-4c4b-a366-9df35cf8db54
# ╠═751a2de4-aa17-425b-b23e-0842aee30c44
# ╠═ff300623-cc98-4d79-adac-766d60d8b4da
# ╠═e935825e-4671-4a7c-b0d0-c1deb6e036ff
# ╠═e64f5bf3-acd3-451c-9b04-bc5a80f08a28
# ╠═cd1a937c-7951-4bdc-b48a-da490fcbc25d
# ╠═29b023e3-879d-438e-bef8-f290bf3d4a55
# ╠═46158577-9672-44f8-bcf3-57eec228d5b0
# ╟─64808ea4-746a-45ec-8f3f-a0625a1816db
# ╠═a64bb46f-189d-466b-83b2-0f17602f5a46
# ╠═69182965-21a3-442a-971e-2e27840a658e
# ╠═f1611d14-74d9-4372-b203-53973cef93bf
# ╟─eafc5a84-7da1-11eb-03d9-29a3cac35040
# ╠═193aee62-7da2-11eb-2ff6-d5562c2a1331
# ╠═82cfa81f-a990-4c55-b05b-f35c7e3649b0
# ╠═19b5416e-85b4-11eb-04ea-3551dd6da8a8
# ╠═2582b3de-58cf-4949-a1a0-f9d45d9b28ee
# ╠═6ac22f4e-85b5-11eb-0674-211b09bb5e57
# ╟─bb90f310-716f-11eb-30e3-532345b94219
# ╠═c63baab2-716f-11eb-2aa2-295e1380384b
