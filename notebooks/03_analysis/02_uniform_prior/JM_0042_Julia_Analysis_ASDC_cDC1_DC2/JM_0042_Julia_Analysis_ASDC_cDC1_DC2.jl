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
	mkpath(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder))
end

# ╔═╡ 90245560-1bcd-11ec-0ba9-35d3debbbc71
md"# $(notebook_folder_title)"

# ╔═╡ 0ae8b435-940c-4990-816d-6612afc6ad9f
md"## Load HPC results"

# ╔═╡ 060dd87a-d8fd-43a5-adfb-d6c216361afa
results_folders = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_00((1[9])|(2[0-8]))_.+)", readdir(projectdir("notebooks","02_fitting","02_uniform_prior/"))))] |> _[[isfile(projectdir("notebooks", "02_fitting","02_uniform_prior/",j, "results", "logp_3d_mat.jlso")) for j in _]]

# ╔═╡ 405c42dc-da20-4b8f-9fca-0f59833aa78d
results_folders_extended = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_00((29)|(3[0-3]))_.+)", readdir(projectdir("notebooks", "02_fitting","02_uniform_prior/"))))] |> _[[isfile(projectdir("notebooks", "02_fitting","02_uniform_prior/",j, "results", "logp_3d_mat.jlso")) for j in _]]


# ╔═╡ b1d99f61-4103-4e5d-b72c-409815d4e0d0
begin
	loglikehoods = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior/", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders]
	loglikehoods_total = [vcat(sum(j, dims=1)...) for j in loglikehoods]
	loglikehoods_r = [permutedims(j, [2,3,1]) for j in loglikehoods]
	relative_eff_r = [rloo.relative_eff(j) for j in loglikehoods_r]
end

# ╔═╡ 201dea27-c988-43cb-b6f2-728f5574145e
begin
	loglikehoods_extended = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior/", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders_extended]
	loglikehoods_extended_r = [permutedims(j, [2,3,1]) for j in loglikehoods_extended]
	relative_eff_extended_r = [rloo.relative_eff(j) for j in loglikehoods_extended_r]
end

# ╔═╡ 92853579-1cdc-46db-84ed-0b621b138fd9
begin
	model_names= [j.captures[1]*"_"*j.captures[2] for j = match.(r"(Model_[1-5])_.*_(nonpooled|pooled)", results_folders)]
	model_id = [match(r"Model_([1-5])", j).captures[1] for j in model_names]
	model_type = [match(r"Model_[1-5].*_(nonpooled|pooled)", j).captures[1] for j in model_names]
end

# ╔═╡ 3ee4af72-a5c2-4f4f-a4cc-7974fa2e7e52
begin
	model_names_extended= [j.captures[1]*"_"*j.captures[2]*"_extended" for j = match.(r"(Model_[1-5])_.*_extended_(pooled)", results_folders_extended)]
	model_id_extended = [match(r"Model_([1-5])", j).captures[1] for j in model_names_extended]
	model_type_extended = [match(r"Model_[1-5].*_(pooled_extended)", j).captures[1] for j in model_names_extended]
end

# ╔═╡ 53c53c8f-304c-4be7-af29-70496db46d6c
md"## LOO-CV"

# ╔═╡ 61a48055-c83d-4c36-ae84-3bbf4bae5cea
begin
	res_loo_cv = map(x-> ParetoSmooth.psis_loo(x), loglikehoods)
	res_loo_r = [rloo.loo(loglikehoods_r[j],r_eff=relative_eff_r[j]) for j in 1:length(results_folders)]
	res_waic_r = [rloo.waic(loglikehoods_r[j],r_eff=relative_eff_r[j]) for j in 1:length(results_folders)]
end

# ╔═╡ d781e4c7-e8fd-45cd-b6c2-c7dc539c1efb
begin
	res_loo_cv_extended = map(x-> ParetoSmooth.psis_loo(x), loglikehoods_extended)
	res_loo_extended_r = [rloo.loo(loglikehoods_extended_r[j],r_eff=relative_eff_extended_r[j]) for j in 1:length(results_folders_extended)]
	res_waic_extended_r = [rloo.waic(loglikehoods_extended_r[j],r_eff=relative_eff_extended_r[j]) for j in 1:length(results_folders_extended)]
end

# ╔═╡ cc437496-18a4-480f-ad7e-c9933a5cc677
begin
	### informative prior results
	dfs_par_pooled = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior/", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_pooled")]]
	## add model and donor
	[dfs_par_pooled[j][!,:model_id] .= model_id[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:model_type] .= model_type[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled)]
end

# ╔═╡ 14b46f70-d9bd-411c-9eed-3fe71527939d
begin
	dfs_par_nonpooled = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior/", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_nonpooled")]]
	for j in 1:length(dfs_par_nonpooled)
		dfs_par_nonpooled[j]= @pipe dfs_par_nonpooled[j] |> 
		combine(_, names(_)[.!map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))].=> (x -> vcat(x...)),
		names(_)[map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))] .=> (x -> repeat(x, 3)), renamecols=false) |> 
		insertcols!(_, :donor=>repeat(["D01","D02", "D04"], outer=Int(nrow(_)/3)))
	end
	## add model and donor
	[dfs_par_nonpooled[j][!,:model_id] .= model_id[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]
	[dfs_par_nonpooled[j][!,:model_type] .= model_type[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]
end

# ╔═╡ 2fd13a67-6187-40fc-9685-6e28f09b140e
begin
	dfs_par_pooled_extended = [JLSO.load(projectdir("notebooks", "02_fitting","02_uniform_prior/", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_extended]
	## add model and donor
	[dfs_par_pooled_extended[j][!,:model_id] .=model_id_extended[j] for j in 1:length(dfs_par_pooled_extended)]
	[dfs_par_pooled_extended[j][!,:model_type] .=model_type_extended[j] for j in 1:length(dfs_par_pooled_extended)]

	[dfs_par_pooled_extended[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled_extended)]
end

# ╔═╡ 56c9a3d9-d72c-47ab-b9a0-f48e1dbed000
begin
	# combine all together
	df_par = @pipe vcat(vcat(dfs_par_pooled...),
	vcat(dfs_par_nonpooled...), 
	vcat(dfs_par_pooled_extended...)) |>
	transform(_,[:δ_ASDCbm, :λ_ASDC, :Δ_cDC1bm, :Δ_DC2bm] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_bm,
	[:δ_cDC1bm, :λ_cDC1] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_bm,
	[:δ_DC2bm, :λ_DC2] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_DC2_bm,
	[:δ_ASDCb, :Δ_cDC1b, :Δ_DC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_ASDC_b,
	[:δ_cDC1b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_cDC1_b,
	[:δ_DC2b] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_DC2_b) |>
	insertcols!(_, :prior=>"lognormal")

	df_par_all= df_par
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

# ╔═╡ 17fb9aef-7340-407d-9cc7-8ceedd415a54
md"Filter out model 3 and rename model ids:"

# ╔═╡ d7d316cd-3e89-4846-8369-09f875059cb5
begin
	df_par_filtered = @pipe df_par |>
	subset(_, :model_id => (x -> x .!= "3")) |>
	transform(_, :model_id => (x-> replace.(replace.(x, "4"=> "3"), "5"=> "4")), renamecols=false)
end

# ╔═╡ 0a88451b-69ff-4a6c-b570-c87d4e99c9ab
md"### Save paremeter estimates and posterior summary stats"

# ╔═╡ 7ab996fe-1ae6-4068-b4e0-18e3f77cfa39
@pipe df_par_filtered |>
_.model_type |>
unique(_)

# ╔═╡ 9abb0a6b-5238-4a76-a86a-e904b48757b6
begin
	# save model estimates for model 1,2,3,4	
	for l in 1:4
		@pipe df_par_filtered |>
		subset(_, :model_id => (x -> x .== string(l))) |>
		subset(_, :model_type => ((x) -> x .∈ Ref(["pooled_extended"]))) |>
		select(_, Not(:prior)) |>
		select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
		groupby(_, [:model_id, :model_type, :donor]) |>
		combine(_, Symbol.(names(_)[names(_) .∉ Ref(["model_id", "model_type", "donor", "prior"])]) .=> (x -> [[median(x), [MCMCChains._hpd(convert.(Float64,x); alpha=0.2)...]...]]), renamecols=false) |>
		DataFrames.stack(_, Not([:model_id, :model_type, :donor])) |> 
		transform(_, :value => ByRow(x -> (median=x[1], ci_80_l = x[2], ci_80_u=x[3])) => AsTable)|>
		select(_, Not(:value)) |> 
		sort(_, :model_id) |>
		transform(_,:model_id => (x -> tryparse.(Int,x) ), renamecols=false)|>
		transform(_, :model_type => (x -> string.(x)), renamecols=false)|>
		rename(_, :variable => :parameter) |> 
		transform(_, :parameter => (x -> string.(x)), renamecols=false) |>
		save(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "Parameter_posterior_summary_stats_model_"*string(l)*".csv"), _)

		@pipe df_par_filtered |>
		subset(_, :model_id => (x -> x .== string(l))) |>
		subset(_, :model_type => ((x) -> x .∈ Ref(["pooled_extended"]))) |>
		select(_, Not(:prior)) |>
		select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
		save(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "Parameter_full_posterior_model_"*string(l)*".csv"), _)
	end
end

# ╔═╡ 8b55a586-4464-4a7d-a315-0229f53546f5
md"### Plot model comparison"

# ╔═╡ a09a4993-f6b8-492c-a101-fb95f660e6c5
begin
	#save elpd differences
	save(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.csv"), df_arviz_loo)
	save(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.csv"), df_arviz_lop)
	save(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.csv"), df_arviz_loo_extended)
	save(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.csv"), df_arviz_lop_extended)
end

# ╔═╡ 99db6e93-5ec4-4a60-bb26-cbabef78793e
begin
	#save elpd differences plot
	p_compare_lop = ArviZ.plot_compare(df_arviz_lop,insample_dev=false)
	p_compare_lop.set_title("Leave-one-population-out PSIS-LOO-CV")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset.svg"))

	p_compare_loo= ArviZ.plot_compare(df_arviz_loo,insample_dev=false)
	p_compare_loo.set_title("Leave-one-out PSIS-LOO-CV")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample.svg"))

	p_compare_lop_extended = ArviZ.plot_compare(df_arviz_lop_extended,insample_dev=false)
	p_compare_lop_extended.set_title("Leave-one-population-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.svg"))

	p_compare_loo_extended = ArviZ.plot_compare(df_arviz_loo_extended,insample_dev=false)
	p_compare_loo_extended.set_title("Leave-one-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis","02_uniform_prior/", notebook_folder, "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.svg"))
end

# ╔═╡ ccce3cde-3f0e-463c-a7fd-bf913aa1c25c
md"## Libraries"

# ╔═╡ Cell order:
# ╠═90245560-1bcd-11ec-0ba9-35d3debbbc71
# ╠═fb9f3e12-294d-42de-a8f1-b673d320e845
# ╠═0ae8b435-940c-4990-816d-6612afc6ad9f
# ╠═060dd87a-d8fd-43a5-adfb-d6c216361afa
# ╠═405c42dc-da20-4b8f-9fca-0f59833aa78d
# ╠═b1d99f61-4103-4e5d-b72c-409815d4e0d0
# ╠═201dea27-c988-43cb-b6f2-728f5574145e
# ╠═92853579-1cdc-46db-84ed-0b621b138fd9
# ╠═3ee4af72-a5c2-4f4f-a4cc-7974fa2e7e52
# ╠═53c53c8f-304c-4be7-af29-70496db46d6c
# ╠═61a48055-c83d-4c36-ae84-3bbf4bae5cea
# ╠═d781e4c7-e8fd-45cd-b6c2-c7dc539c1efb
# ╠═cc437496-18a4-480f-ad7e-c9933a5cc677
# ╠═14b46f70-d9bd-411c-9eed-3fe71527939d
# ╠═2fd13a67-6187-40fc-9685-6e28f09b140e
# ╠═56c9a3d9-d72c-47ab-b9a0-f48e1dbed000
# ╠═17fb9aef-7340-407d-9cc7-8ceedd415a54
# ╠═d7d316cd-3e89-4846-8369-09f875059cb5
# ╟─0a88451b-69ff-4a6c-b570-c87d4e99c9ab
# ╠═7ab996fe-1ae6-4068-b4e0-18e3f77cfa39
# ╠═9abb0a6b-5238-4a76-a86a-e904b48757b6
# ╟─8b55a586-4464-4a7d-a315-0229f53546f5
# ╠═c2a3b797-a097-4aa7-887f-0a16e437a440
# ╠═a09a4993-f6b8-492c-a101-fb95f660e6c5
# ╠═99db6e93-5ec4-4a60-bb26-cbabef78793e
# ╟─ccce3cde-3f0e-463c-a7fd-bf913aa1c25c
# ╠═c8cca440-0b47-4d97-9bfd-23768de0046a
# ╠═8d8c01bd-fbc0-4ee1-b9b7-aa9ec0b7581b
# ╠═4f8629ec-74d3-4c53-b3d6-d1947a354771
