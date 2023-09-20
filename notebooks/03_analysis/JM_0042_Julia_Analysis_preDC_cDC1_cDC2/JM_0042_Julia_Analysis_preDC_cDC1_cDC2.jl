### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ c8cca440-0b47-4d97-9bfd-23768de0046a
using DrWatson

# ╔═╡ 4f8629ec-74d3-4c53-b3d6-d1947a354771
begin
	DrWatson.@quickactivate "Model of DC Differentiation"
	import ParetoSmooth
	using Turing
	using DataFrames
	using DataFramesMeta
	using CategoricalArrays
	using Distributions
	using Plots
	using StatsPlots
	using Plots.PlotMeasures
	using Images
	using CSV
	using JLSO
	using MCMCChains
	using DelimitedFiles
	using Pipe: @pipe
	using RCall
	using TexTables
	using ArviZ
	using PyPlot
end

# ╔═╡ fb9f3e12-294d-42de-a8f1-b673d320e845
begin
	@rimport loo as rloo
	notebook_folder_title = basename(@__DIR__)
	notebook_folder = joinpath(basename(@__DIR__), "results")
	mkpath(projectdir("notebooks", "03_analysis", notebook_folder))
	data_folder = "data_derek_20210823"
end

# ╔═╡ 90245560-1bcd-11ec-0ba9-35d3debbbc71
md"# $(notebook_folder_title)"

# ╔═╡ 0ae8b435-940c-4990-816d-6612afc6ad9f
md"## Load HPC results"

# ╔═╡ 405c42dc-da20-4b8f-9fca-0f59833aa78d
begin
	results_folders = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_00((1[9])|(2[0-8]))_.+)", readdir(projectdir("notebooks","02_fitting"))))] |> _[[isfile(projectdir("notebooks", "02_fitting",j, "results", "logp_3d_mat.jlso")) for j in _]]
	
	results_folders_extended = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_00((29)|(3[0-3]))_.+)", readdir(projectdir("notebooks", "02_fitting"))))] |> _[[isfile(projectdir("notebooks", "02_fitting",j, "results", "logp_3d_mat.jlso")) for j in _]]

	# results_folders_uniform = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_02((5[5-9])|(6[0-4]))_.+)", readdir(projectdir("notebooks"))))] |> _[[isfile(projectdir("notebooks",j, "results", "logp_3d_mat.jlso")) for j in _]]
	
	# results_folders_uniform_extended = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_026[5-9]_.+)", readdir(projectdir("notebooks"))))] |> _[[isfile(projectdir("notebooks",j, "results", "logp_3d_mat.jlso")) for j in _]]
end

# ╔═╡ 201dea27-c988-43cb-b6f2-728f5574145e
begin
	loglikehoods = [JLSO.load(projectdir("notebooks", "02_fitting", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders]
	loglikehoods_total = [vcat(sum(j, dims=1)...) for j in loglikehoods]
	loglikehoods_r = [permutedims(j, [2,3,1]) for j in loglikehoods]
	relative_eff_r = [rloo.relative_eff(j) for j in loglikehoods_r]

	loglikehoods_extended = [JLSO.load(projectdir("notebooks", "02_fitting", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders_extended]
	loglikehoods_extended_r = [permutedims(j, [2,3,1]) for j in loglikehoods_extended]
	relative_eff_extended_r = [rloo.relative_eff(j) for j in loglikehoods_extended_r]

	# loglikehoods_uniform = [JLSO.load(projectdir("notebooks",j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders_uniform]
	# loglikehoods_uniform_r = [permutedims(j, [2,3,1]) for j in loglikehoods_uniform]
	# relative_eff_uniform_r = [rloo.relative_eff(j) for j in loglikehoods_uniform_r]

	# loglikehoods_uniform_extended = [JLSO.load(projectdir("notebooks",j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders_uniform_extended]
	# loglikehoods_uniform_extended_r = [permutedims(j, [2,3,1]) for j in loglikehoods_uniform_extended]
	# relative_eff_uniform_extended_r = [rloo.relative_eff(j) for j in loglikehoods_uniform_extended_r]
end

# ╔═╡ 3ee4af72-a5c2-4f4f-a4cc-7974fa2e7e52
begin
	model_names= [j.captures[1]*"_"*j.captures[2] for j = match.(r"(Model_[1-5])_.*_(nonpooled|pooled)", results_folders)]
	model_id = [match(r"Model_([1-5])", j).captures[1] for j in model_names]
	model_type = [match(r"Model_[1-5].*_(nonpooled|pooled)", j).captures[1] for j in model_names]

	model_names_extended= [j.captures[1]*"_"*j.captures[2]*"_extended" for j = match.(r"(Model_[1-5])_.*_extended_(pooled)", results_folders_extended)]
	model_id_extended = [match(r"Model_([1-5])", j).captures[1] for j in model_names_extended]
	model_type_extended = [match(r"Model_[1-5].*_(pooled_extended)", j).captures[1] for j in model_names_extended]

	# model_names_uniform= [j.captures[1]*"_"*j.captures[5] for j = match.(r"(Model_[1-5])_.*((20210823)|(202510823))_(nonpooled|pooled)", results_folders_uniform)]
	# model_id_uniform = [match(r"Model_([1-5])", j).captures[1] for j in model_names_uniform]
	# model_type_uniform = [match(r"Model_[1-5].*_(nonpooled|pooled)", j).captures[1] for j in model_names_uniform]

	# model_names_uniform_extended= [j.captures[1]*"_"*j.captures[2]*"_extended" for j = match.(r"(Model_[1-5])_.*20210823_extended_(pooled)", results_folders_uniform_extended)]
	# model_id_uniform_extended = [match(r"Model_([1-5])", j).captures[1] for j in model_names_uniform_extended]
	# model_type_uniform_extended = [match(r"Model_[1-5].*_(pooled_extended)", j).captures[1] for j in model_names_uniform_extended]
end

# ╔═╡ 53c53c8f-304c-4be7-af29-70496db46d6c
md"## LOO-CV"

# ╔═╡ d781e4c7-e8fd-45cd-b6c2-c7dc539c1efb
begin
	res_loo_cv = map(x-> ParetoSmooth.psis_loo(x), loglikehoods)
	res_loo_r = [rloo.loo(loglikehoods_r[j],r_eff=relative_eff_r[j]) for j in 1:length(results_folders)]
	res_waic_r = [rloo.waic(loglikehoods_r[j],r_eff=relative_eff_r[j]) for j in 1:length(results_folders)]

	res_loo_cv_extended = map(x-> ParetoSmooth.psis_loo(x), loglikehoods_extended)
	res_loo_extended_r = [rloo.loo(loglikehoods_extended_r[j],r_eff=relative_eff_extended_r[j]) for j in 1:length(results_folders_extended)]
	res_waic_extended_r = [rloo.waic(loglikehoods_extended_r[j],r_eff=relative_eff_extended_r[j]) for j in 1:length(results_folders_extended)]

	# res_loo_cv_uniform = map(x-> ParetoSmooth.psis_loo(x), loglikehoods_uniform)
	# res_loo_cv_uniform_extended = map(x-> ParetoSmooth.psis_loo(x), loglikehoods_uniform_extended)
end

# ╔═╡ 56c9a3d9-d72c-47ab-b9a0-f48e1dbed000
begin
	### informative prior results
	dfs_par_pooled = [JLSO.load(projectdir("notebooks", "02_fitting", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_pooled")]]
	## add model and donor
	[dfs_par_pooled[j][!,:model_id] .= model_id[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:model_type] .= model_type[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled)]

	dfs_par_nonpooled = [JLSO.load(projectdir("notebooks", "02_fitting", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_nonpooled")]]
	for j in 1:length(dfs_par_nonpooled)
		dfs_par_nonpooled[j]= @pipe dfs_par_nonpooled[j] |> 
		combine(_, names(_)[.!map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))].=> (x -> vcat(x...)),
		names(_)[map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))] .=> (x -> repeat(x, 3)), renamecols=false) |> 
		insertcols!(_, :donor=>repeat(["C66","C67", "C68"], outer=Int(nrow(_)/3)))
	end
	## add model and donor
	[dfs_par_nonpooled[j][!,:model_id] .= model_id[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]
	[dfs_par_nonpooled[j][!,:model_type] .= model_type[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]



	dfs_par_pooled_extended = [JLSO.load(projectdir("notebooks", "02_fitting", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_extended]
	## add model and donor
	[dfs_par_pooled_extended[j][!,:model_id] .=model_id_extended[j] for j in 1:length(dfs_par_pooled_extended)]
	[dfs_par_pooled_extended[j][!,:model_type] .=model_type_extended[j] for j in 1:length(dfs_par_pooled_extended)]

	[dfs_par_pooled_extended[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled_extended)]




	# combine all together
	df_par = @pipe vcat(vcat(dfs_par_pooled...),
	vcat(dfs_par_nonpooled...), 
	vcat(dfs_par_pooled_extended...)) |>
	transform(_,[:δ_ASDCbm, :λ_ASDC, :Δ_cDC1bm, :Δ_cDC2bm] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_bm,
	[:δ_cDC1bm, :λ_cDC1] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_bm,
	[:δ_cDC2bm, :λ_cDC2] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC2_bm,
	[:δ_ASDCb, :Δ_cDC1b, :Δ_cDC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_b,
	[:δ_cDC1b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_b,
	[:δ_cDC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC2_b) |>
	insertcols!(_, :prior=>"lognormal")



	### uninformative uniform prior results ####################################################
	# dfs_par_pooled = [JLSO.load(projectdir("notebooks", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_uniform[contains.(model_names_uniform,"_pooled")]]
	# ## add model and donor
	# [dfs_par_pooled[j][!,:model_id] .= model_id_uniform[contains.(model_names_uniform,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	# [dfs_par_pooled[j][!,:model_type] .= model_type_uniform[contains.(model_names_uniform,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	# [dfs_par_pooled[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled)]

	# dfs_par_nonpooled = [JLSO.load(projectdir("notebooks", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_uniform[contains.(model_names_uniform,"_nonpooled")]]
	# for j in 1:length(dfs_par_nonpooled)
	# 	dfs_par_nonpooled[j]= @pipe dfs_par_nonpooled[j] |> 
	# 	combine(_, names(_)[.!map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))].=> (x -> vcat(x...)),
	# 	names(_)[map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))] .=> (x -> repeat(x, 3)), renamecols=false) |> 
	# 	insertcols!(_, :donor=>repeat(["C66","C67", "C68"], outer=Int(nrow(_)/3)))
	# end
	# ## add model and donor
	# [dfs_par_nonpooled[j][!,:model_id] .= model_id_uniform[contains.(model_names_uniform,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]
	# [dfs_par_nonpooled[j][!,:model_type] .= model_type_uniform[contains.(model_names_uniform,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]



	# dfs_par_pooled_extended = [JLSO.load(projectdir("notebooks", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_uniform_extended]
	# ## add model and donor
	# [dfs_par_pooled_extended[j][!,:model_id] .=model_id_uniform_extended[j] for j in 1:length(dfs_par_pooled_extended)]
	# [dfs_par_pooled_extended[j][!,:model_type] .=model_type_uniform_extended[j] for j in 1:length(dfs_par_pooled_extended)]

	# [dfs_par_pooled_extended[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled_extended)]




	# # combine all together
	# df_par_uninformative = @pipe vcat(vcat(dfs_par_pooled...),
	# vcat(dfs_par_nonpooled...), 
	# vcat(dfs_par_pooled_extended...)) |>
	# transform(_,[:δ_ASDCbm, :λ_ASDC, :Δ_cDC1bm, :Δ_cDC2bm] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_bm,
	# [:δ_cDC1bm, :λ_cDC1] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_bm,
	# [:δ_cDC2bm, :λ_cDC2] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC2_bm,
	# [:δ_ASDCb, :Δ_cDC1b, :Δ_cDC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_b,
	# [:δ_cDC1b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_b,
	# [:δ_cDC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC2_b) |>
	# insertcols!(_, :prior=>"uniform")

	df_par_all= df_par #vcat(df_par, df_par_uninformative)

	df_par_filtered = @pipe df_par |>
	subset(_, :model_id => (x -> x .!= "3")) |>
	transform(_, :model_id => (x-> replace.(replace.(x, "4"=> "3"), "5"=> "4")), renamecols=false)
end
# ╔═╡ 8b55a586-4464-4a7d-a315-0229f53546f5
md"### Plot model comparison"

# ╔═╡ 9abb0a6b-5238-4a76-a86a-e904b48757b6
begin
	# save model estimates for model 1,2,3,4	
	for l in 1:4
		@pipe df_par_filtered |>
		subset(_, :model_id => (x -> x .== string(l))) |>
		subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
		select(_, Not(:prior)) |>
		select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
		groupby(_, [:model_id, :model_type, :donor]) |>
		combine(_, Symbol.(names(_)[names(_) .∉ Ref(["model_id", "model_type", "donor", "prior"])]) .=> (x -> [[mean(x), [MCMCChains._hpd(convert.(Float64,x); alpha=0.2)...]...]]), renamecols=false) |>
		DataFrames.stack(_, Not([:model_id, :model_type, :donor])) |> 
		transform(_, :value => ByRow(x -> (mean=x[1], ci_80_l = x[2], ci_80_u=x[3])) => AsTable)|>
		select(_, Not(:value)) |> 
		sort(_, :model_id) |>
		transform(_,:model_id => (x -> tryparse.(Int,x) ), renamecols=false)|>
		transform(_, :model_type => (x -> string.(x)), renamecols=false)|>
		rename(_, :variable => :parameter) |> 
		transform(_, :parameter => (x -> string.(x)), renamecols=false) |>
		save(projectdir("notebooks", "03_analysis", notebook_folder, "Parameter_posterior_summary_stats_model_"*string(l)*".csv"), _)

		@pipe df_par_filtered |>
		subset(_, :model_id => (x -> x .== string(l))) |>
		subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
		select(_, Not(:prior)) |>
		select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
		save(projectdir("notebooks", "03_analysis", notebook_folder, "Parameter_full_posterior_model_"*string(l)*".csv"), _)
	end
end

# ╔═╡ c2a3b797-a097-4aa7-887f-0a16e437a440
begin
	res_compare_loo = ParetoSmooth.loo_compare(res_loo_cv[:], model_names=model_names)
	res_compare_loo_r = rloo.loo_compare(res_loo_r...)
	res_compare_waic_r = rloo.loo_compare(res_waic_r...)
	@rput res_compare_loo_r
	@rput res_compare_waic_r

	diff_colnames = rcopy(R"colnames(res_compare_loo_r)")
	diff_rownames = rcopy(R"rownames(res_compare_loo_r)")
	diff_mat = rcopy(R"res_compare_loo_r")

	diff_colnames_waic = rcopy(R"colnames(res_compare_waic_r)")
	diff_rownames_waic = rcopy(R"rownames(res_compare_waic_r)")
	diff_mat_waic = rcopy(R"res_compare_waic_r")

	df_loo_compare= @pipe DataFrame(diff_mat, :auto) |> rename!(_, diff_colnames) |> hcat(_, DataFrame(model=model_names[tryparse.(Int,replace.(diff_rownames, "model"=> ""))])) |> transform(_, :model => (x -> categorical(x, levels=x, compress=true)), renamecols=false)
	df_loo_compare_waic= @pipe DataFrame(diff_mat_waic, :auto) |> rename!(_, diff_colnames_waic) |> hcat(_, DataFrame(model=model_names[tryparse.(Int,replace.(diff_rownames_waic, "model"=> ""))])) |> transform(_, :model => (x -> categorical(x, levels=x, compress=true)), renamecols=false)
	df_loo_compare_jl = @pipe DataFrame(res_compare_loo.estimates) |> subset(_, :statistic => x -> x .== :cv_elpd) |> unstack(_, :statistic, :value) |> leftjoin(_, DataFrame(model=[keys(res_compare_loo.std_err)...], cv_elpd_se=[values(res_compare_loo.std_err)...]), on=:model) |> transform(_, :model => (x -> categorical(string.(x), levels=string.(x), compress=true)), renamecols=false)


	res_compare_loo_extended = ParetoSmooth.loo_compare(res_loo_cv_extended[:], model_names=model_names_extended)
	res_compare_loo_extended_r = rloo.loo_compare(res_loo_extended_r...)
	res_compare_waic_extended_r = rloo.loo_compare(res_waic_extended_r...)
	@rput res_compare_loo_extended_r
	@rput res_compare_waic_extended_r

	diff_colnames_extended = rcopy(R"colnames(res_compare_loo_extended_r)")
	diff_rownames_extended = rcopy(R"rownames(res_compare_loo_extended_r)")
	diff_mat_extended = rcopy(R"res_compare_loo_extended_r")

	diff_colnames_waic_extended = rcopy(R"colnames(res_compare_waic_extended_r)")
	diff_rownames_waic_extended = rcopy(R"rownames(res_compare_waic_extended_r)")
	diff_mat_waic_extended = rcopy(R"res_compare_waic_extended_r")

	df_loo_compare_extended= @pipe DataFrame(diff_mat_extended, :auto) |> rename!(_, diff_colnames_extended) |> hcat(_, DataFrame(model=model_names_extended[tryparse.(Int,replace.(diff_rownames_extended, "model"=> ""))])) |> transform(_, :model => (x -> categorical(x, levels=x, compress=true)), renamecols=false)
	df_loo_compare_extended_waic= @pipe DataFrame(diff_mat_waic_extended, :auto) |> rename!(_, diff_colnames_waic_extended) |> hcat(_, DataFrame(model=model_names_extended[tryparse.(Int,replace.(diff_rownames_waic_extended, "model"=> ""))])) |> transform(_, :model => (x -> categorical(x, levels=x, compress=true)), renamecols=false)
	df_loo_compare_extended_jl = @pipe DataFrame(res_compare_loo_extended.estimates) |> subset(_, :statistic => x -> x .== :cv_elpd) |> unstack(_, :statistic, :value) |> leftjoin(_, DataFrame(model=[keys(res_compare_loo_extended.std_err)...], cv_elpd_se=[values(res_compare_loo_extended.std_err)...]), on=:model) |> transform(_, :model => (x -> categorical(string.(x), levels=string.(x), compress=true)), renamecols=false)

	# res_compare_loo_uniform = ParetoSmooth.loo_compare(res_loo_cv_uniform[:], model_names=model_names_uniform)
	# df_loo_compare_uniform_jl = @pipe DataFrame(res_compare_loo_uniform.estimates) |> subset(_, :statistic => x -> x .== :cv_elpd) |> unstack(_, :statistic, :value) |> leftjoin(_, DataFrame(model=[keys(res_compare_loo_uniform.std_err)...], cv_elpd_se=[values(res_compare_loo_uniform.std_err)...]), on=:model) |> transform(_, :model => (x -> categorical(string.(x), levels=string.(x), compress=true)), renamecols=false)
	# res_compare_loo_uniform_extended = ParetoSmooth.loo_compare(res_loo_cv_uniform_extended[:], model_names=model_names_uniform_extended)
	# df_loo_compare_uniform_extended_jl = @pipe DataFrame(res_compare_loo_uniform_extended.estimates) |> subset(_, :statistic => x -> x .== :cv_elpd) |> unstack(_, :statistic, :value) |> leftjoin(_, DataFrame(model=[keys(res_compare_loo_uniform_extended.std_err)...], cv_elpd_se=[values(res_compare_loo_uniform_extended.std_err)...]), on=:model) |> transform(_, :model => (x -> categorical(string.(x), levels=string.(x), compress=true)), renamecols=false)

	# save(projectdir("notebooks", "03_analysis", notebook_folder, "loo_model_comparison_original_dataset.csv"), df_loo_compare_jl)
	# save(projectdir("notebooks", "03_analysis", notebook_folder, "loo_model_comparison_extended_dataset.csv"), df_loo_compare_extended_jl)

	arr_ifd_arviz_lop = Dict{String, Any}()
	arr_ifd_arviz_loo = Dict{String, Any}() 

	include(srcdir("dataprep.jl"))
	donor_ids = ["C66", "C67", "C68"]
	cell_cycle_approach = 3
	ratio_approach = "1c"
	ratio_summary = "median"

	tau_stop = 3.5/24.0
	bc = 0.73
	label_ps = DataFrame(load(datadir("exp_pro", "labeling_parameters.csv")))
	cell_ratios = DataFrame(load(datadir("exp_pro", "cell_ratios.csv")))
	labelling_data = DataFrame(load(datadir("exp_pro", "labelling_data.csv")))

    data_in = prepare_data_turing(labelling_data, cell_ratios, label_ps, tau_stop; population = ["ASDC", "cDC1", "cDC2"], individual = donor_ids, label_p_names = [:fr,:delta, :frac], ratio_approach=ratio_approach, ratio_summary = ratio_summary, mean_data = true)
	groups_id  = @pipe data_in.df |> select(_, [:population_idx, :individual_idx] => ((x,y) -> tryparse.(Int, string.(x) .* string.(y)))) |> Array(_) |> reshape(_,:)
	unique_group_ids = unique(groups_id)
	groups_idx = [findall(x-> x == j, groups_id) for j in unique_group_ids]

	for k in 1: length(loglikehoods)
		arr_ifd_arviz_lop[model_names[k]] = @pipe df_par |> 
		subset(_,:model_id => (x -> x .== model_id[k]), :model_type => (x -> x .== model_type[k])) |> 
		select(_,Not([:model_id, :model_type,:donor, :prior])) |> 
		select(_, .!map(x -> any(ismissing.(x)), eachcol(_) )) |> 
		(; zip((Symbol(j) for j in names(_)),(_[!,Symbol(j)] for j in names(_)))...) |> 
		from_namedtuple(_; log_likelihood = permutedims(vcat([sum(loglikehoods[k][j,:, :], dims=1) for j in groups_idx]...), [3,2,1]))
		
		arr_ifd_arviz_loo[model_names[k]] = @pipe df_par |> 
		subset(_,:model_id => (x -> x .== model_id[k]), :model_type => (x -> x .== model_type[k])) |> 
		select(_,Not([:model_id, :model_type,:donor, :prior])) |> 
		select(_, .!map(x -> any(ismissing.(x)), eachcol(_) )) |> 
		(; zip((Symbol(j) for j in names(_)),(_[!,Symbol(j)] for j in names(_)))...) |> 
		from_namedtuple(_; log_likelihood = permutedims(loglikehoods[k], [3,2,1]))
	end

	arr_ifd_arviz_lop_pooled = Dict([Pair(replace([replace(j, "_pooled"=> "")], "Model_4" => "Model_3", "Model_5" => "Model_4")[1],arr_ifd_arviz_lop[j]) for j in filter(x -> x .∈ Ref(model_names[[6,7,9,10]]), keys(arr_ifd_arviz_lop))])
	arr_ifd_arviz_loo_pooled = Dict([Pair(replace([replace(j, "_pooled"=> "")], "Model_4" => "Model_3", "Model_5" => "Model_4")[1],arr_ifd_arviz_loo[j]) for j in filter(x -> x .∈ Ref(model_names[[6,7,9,10]]), keys(arr_ifd_arviz_loo))])



	donor_ids = ["C66", "C67", "C68", "C53", "C55"]
    data_in = prepare_data_turing(labelling_data, cell_ratios, label_ps, tau_stop; population = ["ASDC", "cDC1", "cDC2"], individual = donor_ids, label_p_names = [:fr,:delta, :frac], ratio_approach=ratio_approach, ratio_summary = ratio_summary, mean_data = true)
	groups_id  = @pipe data_in.df |> select(_, [:population_idx, :individual_idx] => ((x,y) -> tryparse.(Int, string.(x) .* string.(y)))) |> Array(_) |> reshape(_,:)
	unique_group_ids = unique(groups_id)
	groups_idx = [findall(x-> x == j, groups_id) for j in unique_group_ids]

	arr_ifd_arviz_lop_extended = Dict{String, Any}()
	arr_ifd_arviz_loo_extended = Dict{String, Any}() 

	for k in 1: length(loglikehoods_extended)
		arr_ifd_arviz_lop_extended[model_names_extended[k]] = @pipe df_par |> 
		subset(_,:model_id => (x -> x .== model_id_extended[k]), :model_type => (x -> x .== model_type_extended[k])) |> 
		select(_,Not([:model_id, :model_type,:donor, :prior])) |> 
		select(_, .!map(x -> any(ismissing.(x)), eachcol(_) )) |> 
		(; zip((Symbol(j) for j in names(_)),(_[!,Symbol(j)] for j in names(_)))...) |> 
		from_namedtuple(_; log_likelihood = permutedims(vcat([sum(loglikehoods_extended[k][j,:, :], dims=1) for j in groups_idx]...), [3,2,1]))
		
		arr_ifd_arviz_loo_extended[model_names_extended[k]] = @pipe df_par |> 
		subset(_,:model_id => (x -> x .== model_id_extended[k]), :model_type => (x -> x .== model_type_extended[k])) |> 
		select(_,Not([:model_id, :model_type,:donor, :prior])) |> 
		select(_, .!map(x -> any(ismissing.(x)), eachcol(_) )) |> 
		(; zip((Symbol(j) for j in names(_)),(_[!,Symbol(j)] for j in names(_)))...) |> 
		from_namedtuple(_; log_likelihood = permutedims(loglikehoods_extended[k], [3,2,1]))
	end

	arr_ifd_arviz_lop_extended = Dict([Pair(replace([replace(j, "_pooled_extended"=> "")], "Model_4" => "Model_3", "Model_5" => "Model_4")[1],arr_ifd_arviz_lop_extended[j]) for j in filter(x -> x .∈ Ref(model_names_extended[[1,2,4,5]]), keys(arr_ifd_arviz_lop_extended))])
	arr_ifd_arviz_loo_extended = Dict([Pair(replace([replace(j, "_pooled_extended"=> "")], "Model_4" => "Model_3", "Model_5" => "Model_4")[1],arr_ifd_arviz_loo_extended[j]) for j in filter(x -> x .∈ Ref(model_names_extended[[1,2,4,5]]), keys(arr_ifd_arviz_loo_extended))])
	
	df_arviz_lop = ArviZ.compare(arr_ifd_arviz_lop_pooled, "loo")
	df_arviz_loo = ArviZ.compare(arr_ifd_arviz_loo_pooled, "loo")

	df_arviz_lop_extended = ArviZ.compare(arr_ifd_arviz_lop_extended, "loo")
	df_arviz_loo_extended = ArviZ.compare(arr_ifd_arviz_loo_extended, "loo")

end

# ╔═╡ a09a4993-f6b8-492c-a101-fb95f660e6c5
begin
	#save elpd differences
	save(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.csv"), df_arviz_loo)
	save(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.csv"), df_arviz_lop)
	save(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.csv"), df_arviz_loo_extended)
	save(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.csv"), df_arviz_lop_extended)
end

# ╔═╡ 99db6e93-5ec4-4a60-bb26-cbabef78793e
begin
	#save elpd differences plot
	p_compare_lop = ArviZ.plot_compare(df_arviz_lop,insample_dev=false)
	p_compare_lop.set_title("Leave-one-population-out PSIS-LOO-CV")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.svg"))

	p_compare_loo= ArviZ.plot_compare(df_arviz_loo,insample_dev=false)
	p_compare_loo.set_title("Leave-one-out PSIS-LOO-CV")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.svg"))

	p_compare_lop_extended = ArviZ.plot_compare(df_arviz_lop_extended,insample_dev=false)
	p_compare_lop_extended.set_title("Leave-one-population-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.svg"))

	p_compare_loo_extended = ArviZ.plot_compare(df_arviz_loo_extended,insample_dev=false)
	p_compare_loo_extended.set_title("Leave-one-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.svg"))
end

# ╔═╡ 431f30e6-2cf0-413f-96c3-2c5d6b39534d
begin

end

# ╔═╡ fae12768-3fa0-46f3-8839-b91acbbceb99
md"## Parameter estimation"

# ╔═╡ 3c634684-9c53-4726-8f4e-6aa076c52d41
begin

end

# ╔═╡ 3e5258f3-8c55-4122-bb8f-e0590c47708b
begin
	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled", "pooled"]))) |>
	# begin
	# 	plot(
	# 		groupedboxplot(_.model_id, _.p_ASDCbm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.p_cDC1bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.p_cDC2bm, group=_.donor, outliers=false),
	# 		title=["p ASDC" "p cDC1" "p cDC2"],
	# 		legend=:outertopright,
	# 		layout=(3,1),
	# 		size=(300,900),
	# 		xaxis=45,
	# 		xlabel="model",
	# 		left_margin = 5mm
	# 	)
	# end

	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "proliferation_boxplot_all.pdf"))


	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled", "pooled"]))) |>
	# begin
	# 	plot(
	# 		groupedboxplot(_.model_id, _.dwell_ASDC_bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC1_bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC2_bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_ASDC_b, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC1_b, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC2_b, group=_.donor, outliers=false),
	# 		title=["dwell_ASDC_bm" "dwell_cDC1_bm" "dwell_cDC2_bm" "dwell_ASDC_b" "dwell_cDC1_b" "dwell_cDC2_b"],
	# 		legend=:outertopright,
	# 		layout=(3,2),
	# 		size=(600,900),
	# 		xaxis=45,
	# 		xlabel="model",
	# 		left_margin = 5mm
	# 	)
	# end
	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "dwelltime_boxplot_all.pdf"))

	# @pipe df_par_all |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled", "pooled"]))) |>
	# begin
	# 	plot(
	# 		groupedviolin(_.model_id, _.p_ASDCbm, group=_.donor .* "_" .* _.prior, outliers=false),
	# 		groupedviolin(_.model_id, _.p_cDC1bm, group=_.donor .* "_" .* _.prior, outliers=false),
	# 		groupedviolin(_.model_id, _.p_cDC2bm, group=_.donor .* "_" .* _.prior, outliers=false),
	# 		title=["p ASDC" "p cDC1" "p cDC2"],
	# 		layout=(3,1),
	# 		size=(600,900),
	# 		xaxis=45,
	# 		left_margin = 5mm,
	# 		legend=:outertopright
	# 	)
	# end
end


# ╔═╡ 840a038a-55af-45f0-b35c-65c9ec587696
begin
	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.p_ASDCbm), _.model_id .* "_" .* _.model_type,xerror = map(x -> x[2], _.p_ASDCbm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.p_cDC1bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2], _.p_cDC1bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1],_.p_cDC2bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2],_.p_cDC2bm), group=_.donor, lab="mean"),
	# 		title=["p ASDC" "p cDC1" "p cDC2"],
	# 		legend=:outertopright,
	# 		layout=(3,1),
	# 		size=(300,900),
	# 		xaxis=45,
	# 		left_margin = 15mm
	# 	)
	# end

	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "proliferation_forest_80hdp_pooled.pdf"))

	# @pipe df_par_all |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# subset(_, :prior => (x -> x .== "uniform")) |>
	# select(_, [:model_id,:model_type,:donor,:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.p_ASDCbm), _.model_id .* "_" .* _.model_type,xerror = map(x -> x[2], _.p_ASDCbm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.p_cDC1bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2], _.p_cDC1bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1],_.p_cDC2bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2],_.p_cDC2bm), group=_.donor, lab="mean"),
	# 		title=["p ASDC" "p cDC1" "p cDC2"],
	# 		legend=:outertopright,
	# 		layout=(3,1),
	# 		size=(300,900),
	# 		xaxis=45,
	# 		left_margin = 15mm
	# 	)
	# end

	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "proliferation_forest_80hdp_pooled_uniform.pdf"))

	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.p_ASDCbm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.p_ASDCbm), lab="mean"),
	# 		scatter(map(x -> x[1], _.p_cDC1bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2], _.p_cDC1bm), lab="mean"),
	# 		scatter(map(x -> x[1],_.p_cDC2bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2],_.p_cDC2bm), lab="mean"),
	# 		title=["p ASDC" "p cDC1" "p cDC2"],
	# 		legend=:outertopright,
	# 		layout=(3,1),
	# 		size=(300,900),
	# 		xaxis=45,
	# 		left_margin = 20mm
	# 	)
	# end
	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "proliferation_forest_80hdp_nonpooled.pdf"))


	# ## dwell times
	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:dwell_ASDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_ASDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:dwell_ASDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_ASDC_b,:dwell_cDC1_b,:dwell_cDC2_b] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:dwell_ASDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_ASDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.dwell_ASDC_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_ASDC_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_cDC1_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC2_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_cDC2_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_ASDC_b), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_ASDC_b), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_b), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2], _.dwell_cDC1_b), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1],_.dwell_cDC2_b), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2],_.dwell_cDC2_b), group=_.donor, lab="mean"),
	# 		title=["dwell_ASDC_bm" "dwell_cDC1_bm" "dwell_cDC2_bm" "dwell_ASDC_b" "dwell_cDC1_b" "dwell_cDC2_b"],
	# 		legend=:outertopright,
	# 		layout=(3,2),
	# 		size=(600,900),
	# 		xaxis=45,
	# 		left_margin = 15mm
	# 	)
	# end

	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "dwelltime_forest_80hdp_pooled.pdf"))







	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:dwell_ASDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_ASDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:dwell_ASDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_ASDC_b,:dwell_cDC1_b,:dwell_cDC2_b] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:dwell_ASDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_ASDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.dwell_ASDC_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_ASDC_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_cDC1_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC2_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_cDC2_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_ASDC_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_ASDC_b), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2], _.dwell_cDC1_b), lab="mean"),
	# 		scatter(map(x -> x[1],_.dwell_cDC2_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2],_.dwell_cDC2_b), lab="mean"),
	# 		title=["dwell_ASDC_bm" "dwell_cDC1_bm" "dwell_cDC2_bm" "dwell_ASDC_b" "dwell_cDC1_b" "dwell_cDC2_b"],
	# 		legend=:outertopright,
	# 		layout=(3,2),
	# 		size=(600,900),
	# 		xaxis=45,
	# 		left_margin = 20mm
	# 	)
	# end
	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "dwelltime_forest_80hdp_nonpooled.pdf"))


	# function TexTables.regtable(df::DataFrames.DataFrame)
	# 	transform!(df, names(df, Symbol) .=> ByRow(String), renamecols=false)
		
	# 	colnames = names(df)
	# 	cols = map(icol -> TexTables.TableCol(colnames[icol], collect(1:nrow(df)), df[!,colnames[icol]]), 1:ncol(df))
	# 	TexTables.regtable(cols...)
	# end

	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), [MCMCChains._hpd(x; alpha=0.2)...]...]]) .=> [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# DataFrames.stack(_, [:p_ASDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# transform(_, :value => ByRow(x -> (mean=x[1], ci_80_l = x[2], ci_80_u=x[3])) => AsTable)|>
	# select(_, Not(:value)) |> 
	# sort(_, :model_id) |>
	# transform(_,:model_id => (x -> tryparse.(Int,x) ), renamecols=false)|>
	# transform(_, :model_type => (x -> string.(x)), renamecols=false)|>
	# rename(_, :variable => :population) |>
	# transform(_, :population => (x -> string.(x)), renamecols=false) |>
	# transform(_, :population => ByRow(x -> (parameter="p", population=replace(x, "p_"=> "")))=> AsTable) |>
	# _
	# write_tex(joinpath(projectdir("notebooks",notebook_folder_title),"table_tmp.tex"),regtable(_))

	
	# cd(projectdir("notebooks",notebook_folder_title))
	# run(`pdflatex '\documentclass{article}\pagestyle{empty}\usepackage{booktabs}\begin{document}\input{table_tmp}\end{document}'`)
	# run(`pdfcrop texput.pdf results/hdi_table_pooled_p_rate.pdf`)


	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .== "1")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, Not(:prior)) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, Symbol.(names(_)[names(_) .∉ Ref(["model_id", "model_type", "donor", "prior"])]) .=> (x -> [[mean(x), [MCMCChains._hpd(convert.(Float64,x); alpha=0.2)...]...]]), renamecols=false) |>
	# DataFrames.stack(_, Not([:model_id, :model_type, :donor])) |> 
	# transform(_, :value => ByRow(x -> (mean=x[1], ci_80_l = x[2], ci_80_u=x[3])) => AsTable)|>
	# select(_, Not(:value)) |> 
	# sort(_, :model_id) |>
	# transform(_,:model_id => (x -> tryparse.(Int,x) ), renamecols=false)|>
	# transform(_, :model_type => (x -> string.(x)), renamecols=false)|>
	# rename(_, :variable => :parameter) |> 
	# transform(_, :parameter => (x -> string.(x)), renamecols=false) |>
	# transform(_, :parameter => ByRow(x -> "\$" * x * "\$"), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "δ"=>"\\delta")), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "Δ"=>"\\Delta")), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "λ"=>"\\lambda")), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "_"=> "\\_")), renamecols=false) |>
	# begin
	# 	# save(projectdir("notebooks", "03_analysis", notebook_folder, "hdi_table_pooled_parameters_model_1.csv"), _)
	# 	cd(projectdir("notebooks",notebook_folder_title))

	# 	write_tex(joinpath(projectdir("notebooks",notebook_folder_title),"table_tmp.tex"),regtable(_))

	# 	run(`pdflatex '\documentclass{article}\pagestyle{empty}\usepackage{booktabs}\begin{document}\input{table_tmp}\end{document}'`)
	# 	run(`pdfcrop texput.pdf results/hdi_table_pooled_parameters_model_1.pdf`)
	# end

	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .== "2")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, Not(:prior)) |>
	# select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, Symbol.(names(_)[names(_) .∉ Ref(["model_id", "model_type", "donor", "prior"])]) .=> (x -> [[mean(x), [MCMCChains._hpd(convert.(Float64,x); alpha=0.2)...]...]]), renamecols=false) |>
	# DataFrames.stack(_, Not([:model_id, :model_type, :donor])) |> 
	# transform(_, :value => ByRow(x -> (mean=x[1], ci_80_l = x[2], ci_80_u=x[3])) => AsTable)|>
	# select(_, Not(:value)) |> 
	# sort(_, :model_id) |>
	# transform(_,:model_id => (x -> tryparse.(Int,x) ), renamecols=false)|>
	# transform(_, :model_type => (x -> string.(x)), renamecols=false)|>
	# rename(_, :variable => :parameter) |> 
	# transform(_, :parameter => (x -> string.(x)), renamecols=false) |>
	# transform(_, :parameter => ByRow(x -> "\$" * x * "\$"), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "δ"=>"\\delta")), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "Δ"=>"\\Delta")), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "λ"=>"\\lambda")), renamecols=false) |>
	# transform(_, :parameter => (x -> replace.(x, "_"=> "\\_")), renamecols=false) |>
	# begin
	# 	# save(projectdir("notebooks", "03_analysis", notebook_folder, "hdi_table_pooled_parameters_model_2.csv"), _)

	# 	cd(projectdir("notebooks",notebook_folder_title))

	# 	write_tex(joinpath(projectdir("notebooks",notebook_folder_title),"table_tmp.tex"),regtable(_))

	# 	run(`pdflatex '\documentclass{article}\pagestyle{empty}\usepackage{booktabs}\begin{document}\input{table_tmp}\end{document}'`)
	# 	run(`pdfcrop texput.pdf results/hdi_table_pooled_parameters_model_2.pdf`)	
	# end

end

# ╔═╡ 8d8c01bd-fbc0-4ee1-b9b7-aa9ec0b7581b
DrWatson.@quickactivate "Model of DC Differentiation"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataFramesMeta = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DrWatson = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
JLSO = "9da8a3cd-07a3-59c0-a743-3fdc52c30d11"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
ParetoSmooth = "a68b5a21-f429-434e-8bfa-46b447300aac"
Pipe = "b98c9c47-44ae-5843-9183-064241ee97a0"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
CSV = "~0.9.4"
DataFrames = "~1.2.2"
DataFramesMeta = "~0.9.1"
Distributions = "~0.25.16"
DrWatson = "~2.5.0"
Images = "~0.24.1"
JLSO = "~2.6.0"
MCMCChains = "~5.0.1"
ParetoSmooth = "~0.2.0"
Pipe = "~1.3.0"
Plots = "~1.22.1"
StatsPlots = "~0.14.27"
Turing = "~0.18.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "db0a7ff3fbd987055c43b4e12d2fa30aaae8749c"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "3.2.1"

[[AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "15f34cc635546ac072d03fc2cc10083adb4df680"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.2.0"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "c71d9da0b0e5183a3410066e6b85670b0ae2bdbe"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.1"

[[AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "0e09520d3e1b8601cdfc5149672337a45d86025b"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.5"

[[AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "06da6c283cf17cf0f97ed2c07c29b6333ee83dc9"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.2.4"

[[AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "b8d49c34c3da35f220e7295659cd0bab8e739fed"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.33"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[AxisKeys]]
deps = ["AbstractFFTs", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "8b382307c6195762a5473ba3522a2830c3014620"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.1.19"

[[BSON]]
git-tree-sha1 = "92b8a8479128367aaab2620b8e73dff632f5ae69"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.3"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "0ad226aa72d8671f20d0316e03028f0ba1624307"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.32"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "Functors", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "MappedArrays", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "dca5e02c9426b2f8ce86d8e723d0702ff33df234"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.9.8"

[[BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "652aab0fc0d6d4db4cc726425cadf700e9f473f1"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.0"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CPUSummary]]
deps = ["Hwloc", "IfElse", "Static"]
git-tree-sha1 = "ed720e2622820bf584d4ad90e6fcb93d95170b44"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.1.3"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "3a877c2fc5c9b88ed7259fd0bdb7691aad6b50dc"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.4"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[Chain]]
git-tree-sha1 = "cac464e71767e8a04ceee82a889ca56502795705"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.8"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "d88340ab502af66cfffc821e70ae72f7dbdce645"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.11.5"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bd4afa1fdeec0c8b89dad3c6e92bc6e3b0fec9ce"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.6.0"

[[CloseOpenIntervals]]
deps = ["ArrayInterface", "Static"]
git-tree-sha1 = "ce9c0d07ed6e1a4fecd2df6ace144cbd29ba6f37"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.2"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "45efb332df2e86f2cb2e992239b6267d97c9e0b6"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "1a90210acd935f222ea19657f143004d2c2a1117"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.38.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6d1c23e740a586955645500bbec662476204a52c"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.1"

[[CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "bc3930158d2be029e90b7c40d1371c4f54fa04db"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.6"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataFramesMeta]]
deps = ["Chain", "DataFrames", "MacroTools", "Reexport"]
git-tree-sha1 = "29e71b438935977f8905c0cb3a8a84475fc70101"
uuid = "1313f7d8-7da2-5740-9ea0-a2ca25f37964"
version = "0.9.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "f4efaa4b5157e0cdb8283ae0b5428bc9208436ed"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.16"

[[DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "e1703f8c9ec58c7f6a4e97a811079c31cbbb7168"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.31"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DrWatson]]
deps = ["Dates", "FileIO", "LibGit2", "MacroTools", "Pkg", "Random", "Requires", "UnPack"]
git-tree-sha1 = "d6aa02ad618cf31af9bbbf87f87baad632538211"
uuid = "634d3b9d-ee7a-5ddf-bec9-22491ea816e1"
version = "2.5.0"

[[DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "BangBang", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "Random", "ZygoteRules"]
git-tree-sha1 = "532397f64ad49472fb60e328369ecd5dedeff02f"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.15.1"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "254182080498cce7ae4bc863d23bf27c632688f7"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.4.4"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "70a0cfd9b1c86b0209e38fbfe6d8231fd606eeaf"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.1"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[FilePathsBase]]
deps = ["Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "6d4b609786127030d09e6b1ee0e2044ec20eb403"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.11"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7f6ad1a7f4621b4ab8e554133dade99ebc6e7221"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.5"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
git-tree-sha1 = "e2727f02325451f6b24445cd83bfa9aaac19cbe7"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.5"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "3169c8b31863f9a409be1d17693751314241e3eb"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.4"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageAxes]]
deps = ["AxisArrays", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "794ad1d922c432082bc1aaa9fa8ffbd1fe74e621"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.9"

[[ImageContrastAdjustment]]
deps = ["ColorVectorSpace", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "2e6084db6cccab11fe0bc3e4130bd3d117092ed9"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.7"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageDistances]]
deps = ["ColorVectorSpace", "Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6378c34a3c3a216235210d19b9f495ecfff2f85f"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.13"

[[ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[ImageMetadata]]
deps = ["AxisArrays", "ColorVectorSpace", "ImageAxes", "ImageCore", "IndirectArrays"]
git-tree-sha1 = "ae76038347dc4edcdb06b541595268fca65b6a42"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.5"

[[ImageMorphology]]
deps = ["ColorVectorSpace", "ImageCore", "LinearAlgebra", "TiledIteration"]
git-tree-sha1 = "68e7cbcd7dfaa3c2f74b0a8ab3066f5de8f2b71d"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.2.11"

[[ImageQualityIndexes]]
deps = ["ColorVectorSpace", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1198f85fa2481a3bb94bf937495ba1916f12b533"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.2.2"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[Images]]
deps = ["AxisArrays", "Base64", "ColorVectorSpace", "FileIO", "Graphics", "ImageAxes", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageShow", "ImageTransformations", "IndirectArrays", "OffsetArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "8b714d5e11c91a0d945717430ec20f9251af4bd2"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.24.1"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InitialValues]]
git-tree-sha1 = "26c8832afd63ac558b98a823265856670d898b6c"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.10"

[[InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1a8c6237e78b714e901e406c096fc8a65528af7d"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JLSO]]
deps = ["BSON", "CodecZlib", "FilePathsBase", "Memento", "Pkg", "Serialization"]
git-tree-sha1 = "e00feb9d56e9e8518e0d60eef4d1040b282771e2"
uuid = "9da8a3cd-07a3-59c0-a743-3fdc52c30d11"
version = "2.6.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static"]
git-tree-sha1 = "d2bda6aa0b03ce6f141a2dc73d0bcb7070131adc"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.3"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LazyStack]]
deps = ["LinearAlgebra", "NamedDims", "OffsetArrays", "Test", "ZygoteRules"]
git-tree-sha1 = "a8bf67afad3f1ee59d367267adb7c44ccac7fdee"
uuid = "1fad7336-0346-5a1a-a56f-a06ba010965b"
version = "0.0.7"

[[LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "71be1eb5ad19cb4f61fa8c73395c0338fd092ae0"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libtask]]
deps = ["Libtask_jll", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "90c6ed7f9ac449cddacd80d5c1fca59c97d203e7"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.5.3"

[[Libtask_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "901fc8752bbc527a6006a951716d661baa9d54e9"
uuid = "3ae2931a-708c-5973-9c38-ccf7496fb450"
version = "0.4.3+0"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "Requires", "SLEEFPirates", "Static", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "d8e21a5965cc6416b9e87e474cf5fc54e9ab3cff"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.76"

[[MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "04c3fd6da28ebd305120ffb05f0a3b8f9baced0a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.0.1"

[[MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "f3f0c23f0ebe11db62ff1e81412919cf7739053d"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.1"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "91ef121a2c458806973c8aaeb502c57b2f0d74b3"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.3.2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[ManualMemory]]
git-tree-sha1 = "9cb207b18148b2199db259adfa923b45593fe08e"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.6"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Memento]]
deps = ["Dates", "Distributed", "JSON", "Serialization", "Sockets", "Syslogs", "Test", "TimeZones", "UUIDs"]
git-tree-sha1 = "19650888f97362a2ae6c84f0f5f6cda84c30ac38"
uuid = "f28f55f0-a522-5efc-85c2-fe41dfb9b2d9"
version = "1.2.0"

[[MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "4f65bdbbe93475f6ff9ea6969b21532f88d359be"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "5203a4532ad28c44f82c76634ad621d7c90abcbd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.29"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "fb4530603a1e62aa5ed7569f283d4b47c2a92f61"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.38"

[[NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "e9ffc92217b8709e0cf7b8808f6223a4a0936c95"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.11"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "e14c485f6beee0c7a8dcf6128bf70b85f1fe201e"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.9"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[ParetoSmooth]]
deps = ["AxisKeys", "FFTW", "InteractiveUtils", "LinearAlgebra", "LoopVectorization", "MCMCDiagnosticTools", "Statistics", "Tullio"]
git-tree-sha1 = "8e4fe110836107e23289838a7c07aa9ae2ff0b5f"
uuid = "a68b5a21-f429-434e-8bfa-46b447300aac"
version = "0.2.0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "438d35d2d95ae2c5e8780b330592b6de8494e779"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.3"

[[Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "4c2637482176b1c2fb99af4d83cb2ff0328fc33c"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.1"

[[Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "74d358e649e0450cb5d3ff54ca7c8d806ed62765"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.5.1"

[[PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "371a19bb801c1b420b29141750f3a34d6c6634b9"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.1.0"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "00bede2eb099dcc1ddc3f9ec02180c326b420ee2"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.17.2"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "575c18c6b00ce409f75d96fefe33ebe01575457a"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.4"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[Rotations]]
deps = ["LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "2ed8d8a16d703f900168822d83699b8c3c1a5cd8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "2e8150c7d2a14ac68537c7aac25faa6577aff046"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.27"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "91e29a2bb257a4b992c48f35084064578b87d364"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.19.0"

[[ScientificTypesBase]]
git-tree-sha1 = "9c1a0dea3b442024c54ca6a318e8acf842eab06f"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.2.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ad42c30a6204c74d264692e633133dcea0e8b14e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.2"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "a8f30abc7c64a39d389680b74e749cf33f872a70"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.3"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "46d7ccc7104860c38b11966dd1f72ff042f382e4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.10"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "233bc83194181b07b052b4ee24515564b893faf6"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.27"

[[StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "Requires", "SIMDTypes", "Static", "ThreadingUtilities"]
git-tree-sha1 = "1258e25e171aec339866f283a11e7d75867e77d7"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.2.4"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Syslogs]]
deps = ["Printf", "Sockets"]
git-tree-sha1 = "46badfcc7c6e74535cc7d833a91f4ac4f805f86d"
uuid = "cea106d9-e007-5e6c-ad93-58fe2094e9c4"
version = "0.3.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "d620a061cb2a56930b52bdf5cf908a5c4fa8e76a"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.4"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "03013c6ae7f1824131b2ae2fc1d49793b51e8394"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.6"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "632a8d4dbbad6627a4d2d21b1c6ebcaeebb1e1ed"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.2"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "52c5f816857bfb3291c7d25420b1f4aca0a74d18"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.0"

[[TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "6c9040665b2da00d30143261aea22c7427aada1c"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.7"

[[Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "dec7b7839f23efe21770b3b1307ca77c13ed631d"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.66"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "ed55426a514db35f58d36c3812aae89cfc057401"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.6"

[[Tullio]]
deps = ["DiffRules", "LinearAlgebra", "Requires"]
git-tree-sha1 = "7201bbb4c138c18bf14511c4cc8daeac6a52c148"
uuid = "bc48ee85-29a4-5162-ae0b-a64e1601d4bc"
version = "0.2.14"

[[Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "e22a11c2029137b35adf00a0e4842707c653938c"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.18.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "Hwloc", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static"]
git-tree-sha1 = "a5324cccb9ebab2e8bfc9bb8eb684394de2517e1"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.9"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WeakRefStrings]]
deps = ["DataAPI", "Parsers"]
git-tree-sha1 = "4a4cfb1ae5f26202db4f0320ac9344b3372136b0"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.3.0"

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╠═90245560-1bcd-11ec-0ba9-35d3debbbc71
# ╠═fb9f3e12-294d-42de-a8f1-b673d320e845
# ╠═0ae8b435-940c-4990-816d-6612afc6ad9f
# ╠═405c42dc-da20-4b8f-9fca-0f59833aa78d
# ╠═201dea27-c988-43cb-b6f2-728f5574145e
# ╠═3ee4af72-a5c2-4f4f-a4cc-7974fa2e7e52
# ╠═53c53c8f-304c-4be7-af29-70496db46d6c
# ╠═d781e4c7-e8fd-45cd-b6c2-c7dc539c1efb
# ╠═56c9a3d9-d72c-47ab-b9a0-f48e1dbed000
# ╠═8b55a586-4464-4a7d-a315-0229f53546f5
# ╠═9abb0a6b-5238-4a76-a86a-e904b48757b6
# ╠═c2a3b797-a097-4aa7-887f-0a16e437a440
# ╟─a09a4993-f6b8-492c-a101-fb95f660e6c5
# ╟─99db6e93-5ec4-4a60-bb26-cbabef78793e
# ╠═431f30e6-2cf0-413f-96c3-2c5d6b39534d
# ╟─fae12768-3fa0-46f3-8839-b91acbbceb99
# ╠═3c634684-9c53-4726-8f4e-6aa076c52d41
# ╠═3e5258f3-8c55-4122-bb8f-e0590c47708b
# ╠═840a038a-55af-45f0-b35c-65c9ec587696
# ╠═c8cca440-0b47-4d97-9bfd-23768de0046a
# ╠═8d8c01bd-fbc0-4ee1-b9b7-aa9ec0b7581b
# ╠═4f8629ec-74d3-4c53-b3d6-d1947a354771
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
