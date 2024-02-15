### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ c8cca440-0b47-4d97-9bfd-23768de0046a
using DrWatson

# ╔═╡ 8d8c01bd-fbc0-4ee1-b9b7-aa9ec0b7581b
DrWatson.@quickactivate "Model of DC Differentiation"

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
	results_folders = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_00((1[9])|(2[0-8]))_.+)", readdir(projectdir("notebooks","02_fitting","02_uniform_prior"))))] |> _[[isfile(projectdir("notebooks", "02_fitting","02_uniform_prior",j, "results", "logp_3d_mat.jlso")) for j in _]]
	
	results_folders_extended = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_00((29)|(3[0-3]))_.+)", readdir(projectdir("notebooks", "02_fitting","02_uniform_prior"))))] |> _[[isfile(projectdir("notebooks", "02_fitting","02_uniform_prior",j, "results", "logp_3d_mat.jlso")) for j in _]]

	# results_folders_uniform = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_02((5[5-9])|(6[0-4]))_.+)", readdir(projectdir("notebooks"))))] |> _[[isfile(projectdir("notebooks",j, "results", "logp_3d_mat.jlso")) for j in _]]
	
	# results_folders_uniform_extended = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_026[5-9]_.+)", readdir(projectdir("notebooks"))))] |> _[[isfile(projectdir("notebooks",j, "results", "logp_3d_mat.jlso")) for j in _]]
end

# ╔═╡ 201dea27-c988-43cb-b6f2-728f5574145e
begin
	loglikehoods = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders]
	loglikehoods_total = [vcat(sum(j, dims=1)...) for j in loglikehoods]
	loglikehoods_r = [permutedims(j, [2,3,1]) for j in loglikehoods]
	relative_eff_r = [rloo.relative_eff(j) for j in loglikehoods_r]

	loglikehoods_extended = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders_extended]
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
	dfs_par_pooled = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_pooled")]]
	## add model and donor
	[dfs_par_pooled[j][!,:model_id] .= model_id[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:model_type] .= model_type[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled)]

	dfs_par_nonpooled = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_nonpooled")]]
	for j in 1:length(dfs_par_nonpooled)
		dfs_par_nonpooled[j]= @pipe dfs_par_nonpooled[j] |> 
		combine(_, names(_)[.!map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))].=> (x -> vcat(x...)),
		names(_)[map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))] .=> (x -> repeat(x, 3)), renamecols=false) |> 
		insertcols!(_, :donor=>repeat(["D01","D02", "D04"], outer=Int(nrow(_)/3)))
	end
	## add model and donor
	[dfs_par_nonpooled[j][!,:model_id] .= model_id[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]
	[dfs_par_nonpooled[j][!,:model_type] .= model_type[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]



	dfs_par_pooled_extended = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_extended]
	## add model and donor
	[dfs_par_pooled_extended[j][!,:model_id] .=model_id_extended[j] for j in 1:length(dfs_par_pooled_extended)]
	[dfs_par_pooled_extended[j][!,:model_type] .=model_type_extended[j] for j in 1:length(dfs_par_pooled_extended)]

	[dfs_par_pooled_extended[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled_extended)]




	# combine all together
	df_par = @pipe vcat(vcat(dfs_par_pooled...),
	vcat(dfs_par_nonpooled...), 
	vcat(dfs_par_pooled_extended...)) |>
	transform(_,[:δ_ASDCbm, :λ_ASDC, :Δ_cDC1bm, :Δ_DC2bm] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_preDC_bm,
	[:δ_cDC1bm, :λ_cDC1] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_bm,
	[:δ_DC2bm, :λ_DC2] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_DC2_bm,
	[:δ_ASDCb, :Δ_cDC1b, :Δ_DC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_b,
	[:δ_cDC1b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_b,
	[:δ_DC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_DC2_b) |>
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
	# transform(_,[:δ_preDCbm, :λ_preDC, :Δ_cDC1bm, :Δ_cDC2bm] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_preDC_bm,
	# [:δ_cDC1bm, :λ_cDC1] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_bm,
	# [:δ_cDC2bm, :λ_cDC2] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC2_bm,
	# [:δ_preDCb, :Δ_cDC1b, :Δ_cDC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_preDC_b,
	# [:δ_cDC1b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_b,
	# [:δ_cDC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC2_b) |>
	# insertcols!(_, :prior=>"uniform")

	df_par_all= df_par #vcat(df_par, df_par_uninformative)

	df_par_filtered = @pipe df_par |>
	subset(_, :model_id => (x -> x .!= "3")) |>
	transform(_, :model_id => (x-> replace.(replace.(x, "4"=> "3"), "5"=> "4")), renamecols=false)
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
	donor_ids = ["D01", "D02", "D04"]
	cell_cycle_approach = 3
	ratio_approach = "1c"
	ratio_summary = "median"

	tau_stop = 3.5/24.0
	bc = 0.73
	label_ps = DataFrame(load(datadir("exp_pro", "labeling_parameters_revision.csv")))
	cell_ratios = DataFrame(load(datadir("exp_pro", "cell_ratios_revision.csv")))
	labelling_data = DataFrame(load(datadir("exp_pro", "labelling_data_revision.csv")))

    data_in = prepare_data_turing(labelling_data, cell_ratios, label_ps, tau_stop; population = ["ASDC", "cDC1", "DC2"], individual = donor_ids, label_p_names = [:fr,:delta, :frac], ratio_approach=ratio_approach, ratio_summary = ratio_summary, mean_data = true)
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



	donor_ids = ["D01", "D02", "D04", "C66", "C67", "C68", "C55"]
    data_in = prepare_data_turing(labelling_data, cell_ratios, label_ps, tau_stop; population = ["ASDC", "cDC1", "DC2"], individual = donor_ids, label_p_names = [:fr,:delta, :frac], ratio_approach=ratio_approach, ratio_summary = ratio_summary, mean_data = true)
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
		save(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "Parameter_posterior_summary_stats_model_"*string(l)*".csv"), _)

		@pipe df_par_filtered |>
		subset(_, :model_id => (x -> x .== string(l))) |>
		subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
		select(_, Not(:prior)) |>
		select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
		save(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "Parameter_full_posterior_model_"*string(l)*".csv"), _)
	end
end

# ╔═╡ a09a4993-f6b8-492c-a101-fb95f660e6c5
begin
	#save elpd differences
	save(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.csv"), df_arviz_loo)
	save(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.csv"), df_arviz_lop)
	save(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.csv"), df_arviz_loo_extended)
	save(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.csv"), df_arviz_lop_extended)
end

# ╔═╡ 99db6e93-5ec4-4a60-bb26-cbabef78793e
begin
	#save elpd differences plot
	p_compare_lop = ArviZ.plot_compare(df_arviz_lop,insample_dev=false)
	p_compare_lop.set_title("Leave-one-population-out PSIS-LOO-CV")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.svg"))

	p_compare_loo= ArviZ.plot_compare(df_arviz_loo,insample_dev=false)
	p_compare_loo.set_title("Leave-one-out PSIS-LOO-CV")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.svg"))

	p_compare_lop_extended = ArviZ.plot_compare(df_arviz_lop_extended,insample_dev=false)
	p_compare_lop_extended.set_title("Leave-one-population-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.svg"))

	p_compare_loo_extended = ArviZ.plot_compare(df_arviz_loo_extended,insample_dev=false)
	p_compare_loo_extended.set_title("Leave-one-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.svg"))
end

# ╔═╡ fae12768-3fa0-46f3-8839-b91acbbceb99
md"## Parameter estimation"

# ╔═╡ 3e5258f3-8c55-4122-bb8f-e0590c47708b
begin
	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled", "pooled"]))) |>
	# begin
	# 	plot(
	# 		groupedboxplot(_.model_id, _.p_preDCbm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.p_cDC1bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.p_cDC2bm, group=_.donor, outliers=false),
	# 		title=["p preDC" "p cDC1" "p cDC2"],
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
	# 		groupedboxplot(_.model_id, _.dwell_preDC_bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC1_bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC2_bm, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_preDC_b, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC1_b, group=_.donor, outliers=false),
	# 		groupedboxplot(_.model_id, _.dwell_cDC2_b, group=_.donor, outliers=false),
	# 		title=["dwell_preDC_bm" "dwell_cDC1_bm" "dwell_cDC2_bm" "dwell_preDC_b" "dwell_cDC1_b" "dwell_cDC2_b"],
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
	# 		groupedviolin(_.model_id, _.p_preDCbm, group=_.donor .* "_" .* _.prior, outliers=false),
	# 		groupedviolin(_.model_id, _.p_cDC1bm, group=_.donor .* "_" .* _.prior, outliers=false),
	# 		groupedviolin(_.model_id, _.p_cDC2bm, group=_.donor .* "_" .* _.prior, outliers=false),
	# 		title=["p preDC" "p cDC1" "p cDC2"],
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
	# select(_, [:model_id,:model_type,:donor,:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_preDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.p_preDCbm), _.model_id .* "_" .* _.model_type,xerror = map(x -> x[2], _.p_preDCbm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.p_cDC1bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2], _.p_cDC1bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1],_.p_cDC2bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2],_.p_cDC2bm), group=_.donor, lab="mean"),
	# 		title=["p preDC" "p cDC1" "p cDC2"],
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
	# select(_, [:model_id,:model_type,:donor,:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_preDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.p_preDCbm), _.model_id .* "_" .* _.model_type,xerror = map(x -> x[2], _.p_preDCbm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.p_cDC1bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2], _.p_cDC1bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1],_.p_cDC2bm), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2],_.p_cDC2bm), group=_.donor, lab="mean"),
	# 		title=["p preDC" "p cDC1" "p cDC2"],
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
	# select(_, [:model_id,:model_type,:donor,:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_preDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.p_preDCbm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.p_preDCbm), lab="mean"),
	# 		scatter(map(x -> x[1], _.p_cDC1bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2], _.p_cDC1bm), lab="mean"),
	# 		scatter(map(x -> x[1],_.p_cDC2bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2],_.p_cDC2bm), lab="mean"),
	# 		title=["p preDC" "p cDC1" "p cDC2"],
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
	# select(_, [:model_id,:model_type,:donor,:dwell_preDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_preDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:dwell_preDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_preDC_b,:dwell_cDC1_b,:dwell_cDC2_b] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:dwell_preDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_preDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.dwell_preDC_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_preDC_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_cDC1_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC2_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_cDC2_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_preDC_b), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_preDC_b), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_b), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2], _.dwell_cDC1_b), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1],_.dwell_cDC2_b), _.model_id .* "_" .* _.model_type, xerror = map(x -> x[2],_.dwell_cDC2_b), group=_.donor, lab="mean"),
	# 		title=["dwell_preDC_bm" "dwell_cDC1_bm" "dwell_cDC2_bm" "dwell_preDC_b" "dwell_cDC1_b" "dwell_cDC2_b"],
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
	# select(_, [:model_id,:model_type,:donor,:dwell_preDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_preDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:dwell_preDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_preDC_b,:dwell_cDC1_b,:dwell_cDC2_b] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:dwell_preDC_bm,:dwell_cDC1_bm,:dwell_cDC2_bm,:dwell_preDC_b,:dwell_cDC1_b,:dwell_cDC2_b]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.dwell_preDC_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_preDC_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_cDC1_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC2_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_cDC2_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_preDC_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_preDC_b), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_cDC1_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2], _.dwell_cDC1_b), lab="mean"),
	# 		scatter(map(x -> x[1],_.dwell_cDC2_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor, xerror = map(x -> x[2],_.dwell_cDC2_b), lab="mean"),
	# 		title=["dwell_preDC_bm" "dwell_cDC1_bm" "dwell_cDC2_bm" "dwell_preDC_b" "dwell_cDC1_b" "dwell_cDC2_b"],
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
	# select(_, [:model_id,:model_type,:donor,:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_preDCbm, :p_cDC1bm, :p_cDC2bm] .=> (x -> [[mean(x), [MCMCChains._hpd(x; alpha=0.2)...]...]]) .=> [:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
	# DataFrames.stack(_, [:p_preDCbm, :p_cDC1bm, :p_cDC2bm]) |>
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
# ╠═a09a4993-f6b8-492c-a101-fb95f660e6c5
# ╠═99db6e93-5ec4-4a60-bb26-cbabef78793e
# ╟─fae12768-3fa0-46f3-8839-b91acbbceb99
# ╠═3e5258f3-8c55-4122-bb8f-e0590c47708b
# ╠═840a038a-55af-45f0-b35c-65c9ec587696
# ╠═c8cca440-0b47-4d97-9bfd-23768de0046a
# ╠═8d8c01bd-fbc0-4ee1-b9b7-aa9ec0b7581b
# ╠═4f8629ec-74d3-4c53-b3d6-d1947a354771
