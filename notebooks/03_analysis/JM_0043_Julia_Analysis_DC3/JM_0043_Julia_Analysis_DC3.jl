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
end

# ╔═╡ 90245560-1bcd-11ec-0ba9-35d3debbbc71
md"# $(notebook_folder_title)"

# ╔═╡ 0ae8b435-940c-4990-816d-6612afc6ad9f
md"## Load HPC results"

# ╔═╡ 405c42dc-da20-4b8f-9fca-0f59833aa78d
begin
	results_folders = @pipe [try j.captures[1] catch end for j in filter!(p -> p != nothing, match.(r"(JM_004[0-1]_.+)", readdir(projectdir("notebooks", "02_fitting"))))] |> _[[isfile(projectdir("notebooks", "02_fitting",j, "results", "logp_3d_mat.jlso")) for j in _]]
end

# ╔═╡ 201dea27-c988-43cb-b6f2-728f5574145e
begin
	loglikehoods = [JLSO.load(projectdir("notebooks", "02_fitting", j, "results", "logp_3d_mat.jlso"))[:loglike_3d] for j in results_folders]
	loglikehoods_r = [permutedims(j, [2,3,1]) for j in loglikehoods]
	relative_eff_r = [rloo.relative_eff(j) for j in loglikehoods_r]
end

# ╔═╡ 3ee4af72-a5c2-4f4f-a4cc-7974fa2e7e52
begin
	model_names= [j.captures[1]*"_"*j.captures[2] for j = match.(r"(Model_[1-5])_.*_(nonpooled|pooled)", results_folders)]
	model_id = [match(r"Model_([1-5])", j).captures[1] for j in model_names]
	model_type = [match(r"Model_[1-5].*_(nonpooled|pooled)", j).captures[1] for j in model_names]
end

# ╔═╡ 53c53c8f-304c-4be7-af29-70496db46d6c
md"## LOO-CV"

# ╔═╡ d781e4c7-e8fd-45cd-b6c2-c7dc539c1efb
begin
	res_loo_cv = map(x-> ParetoSmooth.psis_loo(x), loglikehoods)
	res_loo_r = [rloo.loo(loglikehoods_r[j],r_eff=relative_eff_r[j]) for j in 1:length(results_folders)]
	res_waic_r = [rloo.waic(loglikehoods_r[j],r_eff=relative_eff_r[j]) for j in 1:length(results_folders)]
end

# ╔═╡ 56c9a3d9-d72c-47ab-b9a0-f48e1dbed000
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

	# save(projectdir("notebooks", "03_analysis", notebook_folder, "loo_model_comparison_original_dataset.csv"), df_loo_compare_jl)
	# save(projectdir("notebooks", "03_analysis", notebook_folder, "loo_model_comparison_extended_dataset.csv"), df_loo_compare_extended_jl)
end

# ╔═╡ 8b55a586-4464-4a7d-a315-0229f53546f5
md"## Parameter estimation"

# ╔═╡ 9abb0a6b-5238-4a76-a86a-e904b48757b6
begin
	### informative prior results
	dfs_par_pooled = [JLSO.load(projectdir("notebooks", "02_fitting", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_pooled")]]
	## add model and donor
	[dfs_par_pooled[j][!,:model_id] .= model_id[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:model_type] .= model_type[contains.(model_names,"_pooled")][j] for j in 1:length(dfs_par_pooled)]
	[dfs_par_pooled[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled)]

	# dfs_par_nonpooled = [JLSO.load(projectdir("notebooks", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders[contains.(model_names,"_nonpooled")]]
	# for j in 1:length(dfs_par_nonpooled)
	# 	dfs_par_nonpooled[j]= @pipe dfs_par_nonpooled[j] |> 
	# 	combine(_, names(_)[.!map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))].=> (x -> vcat(x...)),
	# 	names(_)[map(c -> isa(c, Vector{Union{Missing, Float64}}), eachcol(_))] .=> (x -> repeat(x, 4)), renamecols=false) |> 
	# 	insertcols!(_, :donor=>repeat(["C66","C67", "C68", "C52"], outer=Int(nrow(_)/4)))
	# end
	# ## add model and donor
	# [dfs_par_nonpooled[j][!,:model_id] .= model_id[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]
	# [dfs_par_nonpooled[j][!,:model_type] .= model_type[contains.(model_names,"_nonpooled")][j] for j in 1:length(dfs_par_nonpooled)]



	# dfs_par_pooled_extended = [JLSO.load(projectdir("notebooks", j,"results", "df_mcmc_comp.jlso"))[:df_par_all] for j in results_folders_extended]
	## add model and donor
	# [dfs_par_pooled_extended[j][!,:model_id] .=model_id_extended[j] for j in 1:length(dfs_par_pooled_extended)]
	# [dfs_par_pooled_extended[j][!,:model_type] .=model_type_extended[j] for j in 1:length(dfs_par_pooled_extended)]

	# [dfs_par_pooled_extended[j][!,:donor] .= "All" for j in 1:length(dfs_par_pooled_extended)]

	## mean lifetime calculation for branched process and deterministic delay
	# death = 0.3
	# transition = 0.6
	# death_mean_lifetime = 1/death
	# transition_mean_lifetime = (1/transition)
	# transition_mean_lifetime_w_delay = (1/transition) + Delta
	# weight_death = death/(death+transition)
	# weight_transition = transition/(death+transition)
	# total_mean_lifetime = 1/(death+transition)
	# total_average_mean_lifetime = (death_mean_lifetime*weight_death + transition_mean_lifetime*weight_transition)/2
	# ByRow((x...) -> )
	#	
	function dwell_time(x)
		death = x[1]
		transition = x[2]
		delay = x[3] isa Missing ? 0.0 : x[3]

		return (1/death*(death/(death+transition)) + ((1/transition)+delay)*(transition/(death+transition)))/2
	end
	# combine all together
	# df_par = @pipe vcat(vcat(dfs_par_pooled...),
	# vcat(dfs_par_nonpooled...)) |> 
	# transform(_,[:δ_pDCbm, :λ_pDC, :tau] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_pDC_bm,
	# [:δ_pDCb] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_pDC_b) |>
	# insertcols!(_, :prior=>"lognormal")

	df_par = @pipe vcat(vcat(dfs_par_pooled...)) |> #	vcat(dfs_par_nonpooled...)
	transform(_,[:δ_pDCbm, :λ_pDC, :tau] => ByRow((x...) -> dwell_time(x)) => :dwell_pDC_bm,
	[:δ_pDCb] => ByRow((x...) -> 1/sum(skipmissing(x))) => :dwell_pDC_b) |>
	insertcols!(_, :prior=>"lognormal")
end

# ╔═╡ 99db6e93-5ec4-4a60-bb26-cbabef78793e
begin
	include(srcdir("dataprep.jl"))
	donor_ids = ["C66", "C67", "C68", "C52"]
	cell_cycle_approach = 3
	ratio_approach = "2"
	ratio_summary = "median"
	tau_stop = 3.5/24.0
	bc = 0.73
	label_ps = DataFrame(load(datadir("exp_pro","labeling_parameters.csv")))
	cell_ratios = @linq DataFrame(load(datadir("exp_pro", "cell_ratios.csv"))) |> DataFrames.transform(:approach => (x -> string.(x)) => :approach)
	labelling_data = DataFrame(load(datadir("exp_pro","labelling_data.csv")))
	data_in = prepare_data_turing(labelling_data, cell_ratios, label_ps, tau_stop; population = ["pDC"], individual = donor_ids, ratios = ["R_pDC"], label_p_names = [:fr,:delta, :frac], ratio_approach=ratio_approach, ratio_summary = ratio_summary, mean_data = true)
		
	arr_ifd_arviz_loo = Dict{String, Any}() 

	for k in 1: length(loglikehoods)
		arr_ifd_arviz_loo[model_names[k]] = @pipe df_par |> 
		subset(_,:model_id => (x -> x .== model_id[k]), :model_type => (x -> x .== model_type[k])) |> 
		select(_,Not([:model_id, :model_type,:donor, :prior])) |> 
		select(_, .!map(x -> any(ismissing.(x)), eachcol(_) )) |> 
		(; zip((Symbol(j) for j in DataFrames.names(_)),(_[!,Symbol(j)] for j in DataFrames.names(_)))...) |> 
		from_namedtuple(_; log_likelihood = permutedims(loglikehoods[k], [3,2,1]))
	end

	arr_ifd_arviz_loo_pooled = Dict([Pair(replace(j, "_pooled"=> ""),arr_ifd_arviz_loo[j]) for j in keys(arr_ifd_arviz_loo)])

	df_arviz_loo = ArviZ.compare(arr_ifd_arviz_loo_pooled, "loo")
end

# ╔═╡ c2a3b797-a097-4aa7-887f-0a16e437a440
begin
	## save parameter estimates
	for l in 1:2
		@pipe df_par |> subset(_,:model_id => x -> x .== string(l)) |> 
		select(_, [:p_pDCbm, :δ_pDCbm, :δ_pDCb, :λ_pDC, :tau, :dwell_pDC_bm, :dwell_pDC_b, :σ]) |>
		DataFrames.stack(_) |>
		dropmissing(_) |>
		groupby(_, :variable) |>
		combine(_, :value => (x -> (;mean=mean(x),(;zip((:hpd_l, :hpd_u), MCMCChains._hpd(x))...)...)) => AsTable) |>
		save(projectdir("notebooks", "03_analysis", notebook_folder, "Parameter_posterior_summary_stats_pDC_model_"*string(l)*".csv"), _)

		@pipe df_par |>
		subset(_, :model_id => (x -> x .== string(l))) |>
		select(_, .![any(ismissing.(j)) for j in eachcol(_)]) |>
		save(projectdir("notebooks", "03_analysis", notebook_folder, "Parameter_full_posterior_pDC_model_"*string(l)*".csv"), _)
	end
end

# ╔═╡ a09a4993-f6b8-492c-a101-fb95f660e6c5
md"### Model comparison"

# ╔═╡ 431f30e6-2cf0-413f-96c3-2c5d6b39534d
begin
	## save plots
	p_compare_loo = ArviZ.plot_compare(df_arviz_loo, insample_dev=false)
	p_compare_loo.set_title("Leave-one-out PSIS-LOO-CV (Extended data)")
	gcf()
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_pDC_leave_out_sample.pdf"))
	PyPlot.savefig(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_pDC_leave_out_sample.svg"))
end

# ╔═╡ fae12768-3fa0-46f3-8839-b91acbbceb99
begin
	## save model comparison df
	save(projectdir("notebooks", "03_analysis", notebook_folder, "PSIS_LOO_CV_Model_comparison_pDC_leave_out_sample.csv"), df_arviz_loo)
end

# ╔═╡ 3c634684-9c53-4726-8f4e-6aa076c52d41


# ╔═╡ 3e5258f3-8c55-4122-bb8f-e0590c47708b



# ╔═╡ 840a038a-55af-45f0-b35c-65c9ec587696
begin
	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:p_pDCbm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_pDCbm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_pDCbm]) |>
	# begin
	# 	scatter(map(x -> x[1], _.p_pDCbm), _.model_id .* "_" .* _.model_type,xerror = map(x -> x[2], _.p_pDCbm), group=_.donor, lab="mean",
	# 		title="p pDC",
	# 		legend=:outertopright,
	# 		size=(300,200),
	# 		xaxis=45,
	# 		left_margin = 15mm,
	# 		ylims=(-0.001, length(_.model_id)+0.01),
	# 		tick_direction = :out
	# 	)
	# end

	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "proliferation_forest_80hdp_pooled.pdf"))

	# @pipe df_par |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:p_pDCbm]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:p_pDCbm] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:p_pDCbm]) |>
	# begin
	# 	scatter(map(x -> x[1], _.p_pDCbm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror = map(x -> x[2], _.p_pDCbm), group=_.donor, lab="mean",
	# 		title="p pDC",
	# 		legend=:outertopright,
	# 		size=(400,200),
	# 		xaxis=45,
	# 		left_margin = 15mm,
	# 		ylims=(-0.1, length(_.model_id)+0.1),
	# 		tick_direction = :out
	# 	)
	# end
	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "proliferation_forest_80hdp_nonpooled.pdf"))


	# ## dwell times
	# @pipe df_par_delay |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["pooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:dwell_pDC_bm,:dwell_pDC_b]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:dwell_pDC_bm,:dwell_pDC_b] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:dwell_pDC_bm,:dwell_pDC_b]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.dwell_pDC_bm), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_pDC_bm), group=_.donor, lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_pDC_b), _.model_id .* "_" .* _.model_type,xerror= map(x -> x[2], _.dwell_pDC_b), group=_.donor, lab="mean"),
	# 		title=["dwell_pDC_bm" "dwell_pDC_b"],
	# 		legend=:outertopright,
	# 		layout=(1,2),
	# 		size=(600,200),
	# 		xaxis=45,
	# 		left_margin = 15mm,
	# 		ylims=(-0.001, length(_.model_id)+0.001),
	# 		tick_direction = :out
	# 	)
	# end

	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "dwelltime_forest_80hdp_pooled.pdf"))


	# @pipe df_par_delay |>
	# subset(_, :model_id => (x -> x .!= "4")) |>
	# subset(_, :model_type => ((x) -> x .∈ Ref(["nonpooled"]))) |>
	# select(_, [:model_id,:model_type,:donor,:dwell_pDC_bm,:dwell_pDC_b]) |>
	# groupby(_, [:model_id, :model_type, :donor]) |>
	# combine(_, [:dwell_pDC_bm,:dwell_pDC_b] .=> (x -> [[mean(x), tuple([abs.(MCMCChains._hpd(x; alpha=0.2).-mean(x))...]...)]]) .=> [:dwell_pDC_bm,:dwell_pDC_b]) |>
	# begin
	# 	plot(
	# 		scatter(map(x -> x[1], _.dwell_pDC_bm), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_pDC_bm), lab="mean"),
	# 		scatter(map(x -> x[1], _.dwell_pDC_b), _.model_id .* "_" .* _.model_type .* "_" .* _.donor,xerror= map(x -> x[2], _.dwell_pDC_b), lab="mean"),
	# 		title=["dwell_pDC_bm" "dwell_pDC_b"],
	# 		legend=:outertopright,
	# 		layout=(1,2),
	# 		size=(800,300),
	# 		xaxis=45,
	# 		left_margin = 20mm,
	# 		ylims=(-0.1, length(_.model_id)+0.1),
	# 		tick_direction = :out
	# 	)
	# end
	# savefig(projectdir("notebooks", "03_analysis", notebook_folder, "dwelltime_forest_80hdp_nonpooled.pdf"))


	# @pipe df_par |> subset(_,:model_id => x -> x .== "2") |> 
	# select(_, [:p_pDCbm, :δ_pDCbm, :δ_pDCb, :λ_pDC, :dwell_pDC_bm, :dwell_pDC_b]) |>
	# DataFrames.stack(_) |>
	# groupby(_, :variable) |>
	# combine(_, :value => (x -> (;mean=mean(x),(;zip((:hpd_l, :hpd_u), MCMCChains._hpd(x))...)...)) => AsTable)

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
