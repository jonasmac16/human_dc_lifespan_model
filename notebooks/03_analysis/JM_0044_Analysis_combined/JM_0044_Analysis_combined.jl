### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 5240145d-ba36-479c-bda9-4944e13cf70b
using DrWatson

# ╔═╡ 2ebde24d-634b-467b-8ee0-6d45174f72f3
DrWatson.@quickactivate "Model of DC Differentiation"

# ╔═╡ 69516fb5-c504-4fb7-a633-f3e8c42190a9
begin
	using StatsPlots
	using Distributions
	using CSV
	using DataFrames
	using Latexify
	using Pipe
	using AlgebraOfGraphics
	using CairoMakie
	using CategoricalArrays
	using ColorSchemes
	using JLSO
	using Turing
	using RCall
end

# ╔═╡ 6990831e-caa0-415b-98c6-f031b7653e70
using ArviZ

# ╔═╡ 35909c90-616d-43f4-bfa1-05a1c841b743
using PyPlot

# ╔═╡ 3bbed963-a3f2-4a97-8fe0-a5c435986a48
using PyCall

# ╔═╡ be1a1b97-8cbe-4605-993a-cf8d8a9fa759
using ParetoSmooth

# ╔═╡ f80272ab-7379-4be0-99c8-bd803b0f48ed
include(srcdir("df2latex.jl"))

# ╔═╡ c671ec28-4bc2-11ec-05c2-1952e68a6e4b
begin
	notebook_folder_title = basename(@__DIR__)
	notebook_folder = joinpath(basename(@__DIR__), "results")
	res_folder = projectdir("notebooks","03_analysis",notebook_folder)
	mkpath(res_folder)
end

# ╔═╡ f6a855d5-f852-437b-913b-48a2f1a24469
md"# $(notebook_folder_title)"

# ╔═╡ 8dca14b1-b8db-4eeb-8bf4-af41936e2505
R"library('loo')"

# ╔═╡ e8a0497c-1d9b-465d-a609-ad6fa4f52bec
md"## Normal vs Student's t distribution"

# ╔═╡ 0b015757-d6cb-44c6-94f5-419b46c0dc75
# begin
# 	N = Normal(0.0, 1.0) 
# 	t = [Distributions.LocationScale(0.0, 1.0, TDist(j)) for j in [collect(4:2:16)..., collect(20:20:100)...]]
# end

# ╔═╡ a08682c5-a805-4a82-94f2-a210d48a87bb
begin
# 	p = Plots.plot()
	
# 	for j in t
# 		Plots.plot!(p, j, lab = "t(μ=0.0, σ=1.0, ν="*string(params(j)[3].ν)*")", w=2, alpha=0.75)
# 	end
# 	Plots.plot!(p, N, c=:black, lab="Normal(μ=0.0,σ=1.0)", w=2, alpha=0.75, guidefontsize=12, tickfontsize=12, ylabel="density", grid=false)
# 	savefig(p, joinpath(res_folder, "Normal_v_Student_t.pdf"))
	# p
end

# ╔═╡ ab676ff6-4906-4204-a9f1-f4c7e3c385cc
# begin
# 	fig_dist = CairoMakie.Figure(;resolution=(700,500))
# 	ax_dist = Axis(fig_dist[1,1], ylabel="density")
# 	dist_plot = t
# 	colors_dist = cgrad(:roma, length(dist_plot), categorical=true)
# 	names = "t(μ=0, σ=1, ν=" .*[string(params(j)[3].ν) for j in t] .*")"
	
# 	for (idx, j) in enumerate(dist_plot)
# 		CairoMakie.plot!(ax_dist, j, color = (colors_dist[idx], 0.7),label=names[idx])
# 	end
# 	CairoMakie.plot!(ax_dist, N, color = (:black,1.0),label="N(μ=0.0, σ=1.0)")
	
# 	# ax_dist_label = Axis(fig_dist[1,2])
# 	Legend(fig_dist[1,2], ax_dist, "Distribution")
# 	fig_dist
# end

# ╔═╡ 5b55c1d5-5fdf-462e-bdba-192a589cfc03
# save(joinpath(res_folder,"normal_v_student_makie_thesis.pdf"), fig_dist)

# ╔═╡ 969d39ad-8696-464f-9087-2eb1e886e94e
md"## Load PPC data"

# ╔═╡ 65129028-c29e-4ab4-a154-62a3a7dcdd5d
pooled_results_notebooks = filter(x -> !isnothing(match(r"JM_00(([2][4-9])|([3][0-3]))",x)),readdir(projectdir("notebooks", "02_fitting"), join=true))

# ╔═╡ 4f04a6a7-73fa-42f2-8bbe-ddfaadbbefbd
pooled_results_normal_notebooks = filter(x -> !isnothing(match(r"JM_00(([0][9])|([1][0-8]))",x)),readdir(projectdir("notebooks", "02_fitting"), join=true))

# ╔═╡ bb1f6478-82d5-49cd-8225-51281d911e2d
nonpooled_results_notebooks = filter(x -> !isnothing(match(r"JM_00(([1][9])|([2][0-3]))",x)),readdir(projectdir("notebooks", "02_fitting"), join=true))

# ╔═╡ 268825ca-d8e7-4f1c-ae91-64747a875984
nonpooled_results_normal_notebooks = filter(x -> !isnothing(match(r"JM_000[4-8]",x)),readdir(projectdir("notebooks", "02_fitting"), join=true))

# ╔═╡ e82c870e-8b63-4ddd-9289-2eb7c9e3cd72
pooled_pdc_results_notebooks = filter(x -> !isnothing(match(r"JM_00((3[6,7])|(4[0,1]))",x)),readdir(projectdir("notebooks", "02_fitting"), join=true))

# ╔═╡ 8691540d-c347-4fff-914b-00cea31dc22c
nonpooled_pdc_results_notebooks = filter(x -> !isnothing(match(r"JM_003[4,5,8,9]",x)),readdir(projectdir("notebooks", "02_fitting"), join=true))

# ╔═╡ 8f25ba44-552a-41a3-b7ec-32fe28702964
begin
	model_id_pooled  = [tryparse(Int, match(r"Model_(\d)", j).captures[1]) for j in pooled_results_notebooks]
	
	model_id_nonpooled  = [tryparse(Int, match(r"Model_(\d)", j).captures[1]) for j in nonpooled_results_notebooks]
end

# ╔═╡ 1c20d417-f83d-4210-a17f-f8556b6e74ca
begin
	model_id_normal_pooled  = [tryparse(Int, match(r"Model_(\d)", j).captures[1]) for j in pooled_results_normal_notebooks]
	
	model_id_normal_nonpooled  = [tryparse(Int, match(r"Model_(\d)", j).captures[1]) for j in nonpooled_results_normal_notebooks]
end

# ╔═╡ df7db22c-79c2-4c1e-8f20-cfeb75bcf7c6
begin
	model_id_pdc_pooled  = [tryparse(Int, match(r"Model_(\d)", j).captures[1]) for j in pooled_pdc_results_notebooks]
	
	model_id_pdc_nonpooled  = [tryparse(Int, match(r"Model_(\d)", j).captures[1]) for j in nonpooled_pdc_results_notebooks]
end

# ╔═╡ e128f1e6-b3ef-4d48-97b7-c96248bc84f1
begin
	strata_pooled  = [match(r"_(pooled)", j).captures[1] for j in pooled_results_notebooks]
	strata_nonpooled  = [match(r"_(nonpooled)", j).captures[1] for j in nonpooled_results_notebooks]
end

# ╔═╡ ab51a99b-0c05-4f16-a7c2-97852d83a1e6
begin
	strata_normal_pooled  = [match(r"_(pooled)", j).captures[1] for j in pooled_results_normal_notebooks]
	strata_normal_nonpooled  = [match(r"_(nonpooled)", j).captures[1] for j in nonpooled_results_normal_notebooks]
end

# ╔═╡ d6f0af2f-3449-417e-bf97-5fdbb8646c46
begin
	strata_pdc_pooled  = [match(r"_(pooled)", j).captures[1] for j in pooled_pdc_results_notebooks]
	strata_pdc_nonpooled  = [match(r"_(nonpooled)", j).captures[1] for j in nonpooled_pdc_results_notebooks]
end

# ╔═╡ 8e9b4cfe-25bf-4311-bbb0-e96b183ac8a0
begin
	data_input_pooled  = [isnothing(match(r"_(extended)_pooled", j)) ? "original" : match(r"_(extended)_pooled", j).captures[1] for j in pooled_results_notebooks]
	data_input_nonpooled  = [isnothing(match(r"_(extended)_pooled", j)) ? "original" : match(r"_(extended)_pooled", j).captures[1] for j in nonpooled_results_notebooks]
end

# ╔═╡ caac50f7-8b6a-4940-9d10-9e58a1e31ccb
begin
	data_input_normal_pooled  = [isnothing(match(r"_(extended)_pooled", j)) ? "original" : match(r"_(extended)_pooled", j).captures[1] for j in pooled_results_normal_notebooks]
	data_input_normal_nonpooled  = [isnothing(match(r"_(extended)_pooled", j)) ? "original" : match(r"_(extended)_pooled", j).captures[1] for j in nonpooled_results_normal_notebooks]
end

# ╔═╡ d84a78ea-394c-4151-b5e4-1dff040e634f
begin
	data_input_pdc_pooled  = [isnothing(match(r"_(extended)_pooled", j)) ? "original" : match(r"_(extended)_pooled", j).captures[1] for j in pooled_pdc_results_notebooks]
	data_input_pdc_nonpooled  = [isnothing(match(r"_(extended)_pooled", j)) ? "original" : match(r"_(extended)_pooled", j).captures[1] for j in nonpooled_pdc_results_notebooks]
end

# ╔═╡ a123f0fa-b0c9-4bf3-a2a3-166339bc1999
begin
	priors_pooled = [match(r"\w+_(\w+?)_prior", j).captures[1]  for j in pooled_results_notebooks]
	priors_nonpooled = [match(r"\w+_(\w+?)_prior", j).captures[1] for j in nonpooled_results_notebooks]
end

# ╔═╡ 676788cc-b8ee-47ba-9b9e-2746155132ab
begin
	priors_normal_pooled = [match(r"\w+_(\w+?)_prior", j).captures[1]  for j in pooled_results_normal_notebooks]
	priors_normal_nonpooled = [match(r"\w+_(\w+?)_prior", j).captures[1] for j in nonpooled_results_normal_notebooks]
end

# ╔═╡ 12454b5d-a771-4044-8ef9-ba08ebcb3ff4
begin
	priors_pdc_pooled = [match(r"\w+_(\w+?)_prior", j).captures[1]  for j in pooled_pdc_results_notebooks]
	priors_pdc_nonpooled = [match(r"\w+_(\w+?)_prior", j).captures[1] for j in nonpooled_pdc_results_notebooks]
end

# ╔═╡ 0e40f982-bc34-491f-bdef-143c6b2008cd
begin
	likelihood_nonpooled = [(isnothing(match(r"mean_((student_t)|(studentt))_((lognormal)|(uniform))", j)) ? "normal" : "student_t") for j in nonpooled_results_notebooks]
	likelihood_pooled = [(isnothing(match(r"mean_((student_t)|(studentt))_((lognormal)|(uniform))", j)) ? "normal" : "student_t") for j in pooled_results_notebooks]
	likelihood_pdc_nonpooled = [(isnothing(match(r"mean_((student_t)|(studentt))_((lognormal)|(uniform))", j)) ? "normal" : "student_t") for j in pooled_pdc_results_notebooks]
	likelihood_pdc_pooled = [(isnothing(match(r"mean_((student_t)|(studentt))_((lognormal)|(uniform))", j)) ? "normal" : "student_t") for j in pooled_pdc_results_notebooks]
end	

# ╔═╡ 251e83f8-fb1e-431d-ae95-1d3639a5db2a
begin
	likelihood_normal_pooled = [(isnothing(match(r"mean_((student_t)|(studentt))_((lognormal)|(uniform))", j)) ? "normal" : "student_t") for j in pooled_results_normal_notebooks]
	likelihood_normal_nonpooled = [(isnothing(match(r"mean_((student_t)|(studentt))_((lognormal)|(uniform))", j)) ? "normal" : "student_t") for j in nonpooled_results_normal_notebooks]
end

# ╔═╡ cb6b271e-3595-4ce4-89c6-1fcc2eaa5136
begin
	df=DataFrame()
	
	for (idx, j) in enumerate(pooled_results_notebooks)
		df_tmp = CSV.read(joinpath(j, "results", "df_ppc.csv"), DataFrame)
		insertcols!(df_tmp, :data => data_input_pooled[idx], :strata => strata_pooled[idx], :model => model_id_pooled[idx], :prior=>priors_pooled[idx], :likelihood_f => likelihood_pooled[idx])
		df = vcat(df, df_tmp, cols=:union)
	end
	rename!(df, :timestamp => :time)
	transform!(df, :sample_idx=> x -> categorical(x, levels=unique(x), compress=true), renamecols=false)
	transform!(df, :population=> x -> categorical(x, levels=["ASDC","cDC1", "cDC2"], compress=true), renamecols=false)
	subset!(df, :model => x -> x .∈ Ref([1,2,4,5]))
	transform!(df, :model => (x -> replace(x, 4=> 3, 5 => 4)), renamecols=false)
end

# ╔═╡ dc3227e9-f40f-4b38-9f01-4e479ce75a48
begin
	df_nonpooled=DataFrame()
	
	for (idx, j) in enumerate(nonpooled_results_notebooks)
		df_tmp = CSV.read(joinpath(j, "results", "df_ppc.csv"), DataFrame)
		insertcols!(df_tmp, :data => data_input_nonpooled[idx], :strata => strata_nonpooled[idx], :model => model_id_nonpooled[idx], :prior=>priors_nonpooled[idx], :likelihood_f => likelihood_nonpooled[idx])
		df_nonpooled = vcat(df_nonpooled, df_tmp, cols=:union)
	end
	rename!(df_nonpooled, :timestamp => :time)
	transform!(df_nonpooled, :sample_idx=> x -> categorical(x, levels=unique(x), compress=true), renamecols=false)
	transform!(df_nonpooled, :population=> x -> categorical(x, levels=["ASDC","cDC1", "cDC2"], compress=true), renamecols=false)
	subset!(df_nonpooled, :model => x -> x .∈ Ref([1,2,4,5]))
	transform!(df_nonpooled, :model => (x -> replace(x, 4=> 3, 5 => 4)), renamecols=false)
end

# ╔═╡ 80000554-c2aa-4c96-88d0-c2ac7a452b04
begin
	df_pdc_pooled=DataFrame()
	
	for (idx, j) in enumerate(pooled_pdc_results_notebooks)
		df_tmp = CSV.read(joinpath(j, "results", "df_ppc.csv"), DataFrame)
		insertcols!(df_tmp, :data => data_input_pdc_nonpooled[idx], :strata => strata_pdc_nonpooled[idx], :model => model_id_pdc_nonpooled[idx], :prior=>priors_pdc_nonpooled[idx], :likelihood_f => likelihood_pdc_pooled[idx])
		df_pdc_pooled = vcat(df_pdc_pooled, df_tmp, cols=:union)
	end
	rename!(df_pdc_pooled, :timestamp => :time)
	transform!(df_pdc_pooled, :sample_idx=> x -> categorical(x, levels=unique(x), compress=true), renamecols=false)
end

# ╔═╡ 008601a2-75b8-4114-ac56-8b52f27735c7
begin
	df_pdc_nonpooled=DataFrame()
	
	for (idx, j) in enumerate(nonpooled_pdc_results_notebooks)
		df_tmp = CSV.read(joinpath(j, "results", "df_ppc.csv"), DataFrame)
		insertcols!(df_tmp, :data => data_input_pdc_nonpooled[idx], :strata => strata_pdc_nonpooled[idx], :model => model_id_pdc_nonpooled[idx], :prior=>priors_pdc_nonpooled[idx], :likelihood_f => likelihood_pdc_nonpooled[idx])
		df_pdc_nonpooled = vcat(df_pdc_nonpooled, df_tmp, cols=:union)
	end
	rename!(df_pdc_nonpooled, :timestamp => :time)
	transform!(df_pdc_nonpooled, :sample_idx=> x -> categorical(x, levels=unique(x), compress=true), renamecols=false)
end

# ╔═╡ 08d9207a-7494-4688-90d6-e88548a66da7
md"## Load experimental data"

# ╔═╡ 3fa70e63-ac7b-42e7-87b7-12a0408d0ba6
labelling_data = CSV.read(datadir("exp_pro", "labelling_data.csv"), DataFrame)

# ╔═╡ c3b1af35-75ad-40d6-a192-bc03e1f3da92
labelling_data_mean = @pipe labelling_data |> groupby(_, [:time, :individual,:population]) |> combine(_, :enrichment => (x -> (mean=mean(x), std= std(x))) => AsTable) |> rename(_, :individual => :donor)

# ╔═╡ 7d640dca-c446-408a-bc2e-422345b17da3
md"## Plot PPC data of pooled models"

# ╔═╡ 39aaebec-177c-4171-9c9e-c0df3517b121
set_aog_theme!()

# ╔═╡ 79526fb4-3a88-49fc-994a-1bc44c224ccc
begin
	function plot_ppc_pop(ax, df,df_data,models, data, location, population, donor, prior; alpha=0.5, data_color=:black, max_models=4, color_scheme=cgrad(:roma, max_models, categorical=true)[models])
	
		ppc_df = @pipe df |>
		subset(_, :model => x -> x .∈ Ref(models), :data => x -> x .== data, :location => x -> x .== location, :donor=> x -> x .== donor, :population => x -> x .== population, :prior => x -> x .== prior) |>
		groupby(_, [:time, :model]) |>
		combine(_, :value => (x -> (;zip((:mean, :lower, :upper), Tuple([mean(x),quantile(x,[0.1,0.9])...]))...)) => AsTable) |>
		groupby(_, :model)
		
	
		data_df = @pipe df_data |> subset(_, :donor => x -> x .== donor, :population => x -> x .== population)
		
		band_p = Array{Any, 1}(undef, length(models))
		# scatter_p = Array{Any, 1}(undef, length(models))
		# error_p = Array{Any, 1}(undef, length(models))
		# data_p = Array{Any, 1}(undef, length(models))
		
		# color_scheme = cgrad(color, max_models, categorical=true)[models]#get(colorschemes[color], LinRange(0,1,length(models)))
		
		for (idx, j) in enumerate(ppc_df)
			band_p[idx] = band!(ax,j.time, j.lower, j.upper, transparency=true, color=(color_scheme[idx], alpha), label="model " * string(unique(j.model)[1]))
		end
		
		line_p = [lines!(ax,ppc_df[NamedTuple(keys(ppc_df)[j])].time, ppc_df[NamedTuple(keys(ppc_df)[j])].mean, transparency=true, color=(color_scheme[j], alpha*1.2)) for j in 1:length(keys(ppc_df))]
		
		if (length(data_df.time) > 0) & (location == "b")
			scatter_p = CairoMakie.scatter!(ax,data_df.time, data_df.mean, color=data_color, markersize=10, label=population)
		
			error_p = CairoMakie.errorbars!(ax,data_df.time, data_df.mean, data_df.std, color=data_color, linewidth=1,whiskerwidth = 4)
			res = (band_p, scatter_p)
		else
			res = (band_p, nothing)
		end
	return res
	end
end

# ╔═╡ 56ab98d5-fba1-4644-9b9b-2a1197756dbc
begin
	function plot_predictions(df; donors_plotted = ["C66", "C67", "C68", "C53", "C55"], populations=["ASDC", "cDC1", "cDC2"], dataset="extended", location="b", prior="lognormal", colors=:roma, data_color = [colorant"#755494",colorant"#de3458" ,colorant"#4e65a3"], models=[1,2,4,5], max_models=4, alpha=0.5, f_kwargs = (;), f = CairoMakie.Figure(;f_kwargs...), ax = hcat([[Axis(f[k,j], aspect=1.5) for k in 1:length(donors_plotted)] for j in 1:length(populations)]...))
		
		
		color_scheme=cgrad(colors, max_models, categorical=true, alpha=alpha)[models]

		for (idx ,j) in enumerate(populations)
			ax[1,idx].title= j
		end
		
		bands = Array{Any, 2}(undef, length(donors_plotted), length(populations))
		scatters = Array{Any, 2}(undef, length(donors_plotted), length(populations))

		for (idx, j) in enumerate(donors_plotted)
			
			for (idx2, k) in enumerate(populations)
				bands[idx,idx2], scatters[idx,idx2] = plot_ppc_pop(ax[idx,idx2],df,labelling_data_mean,models, dataset, location, k, j, prior; alpha=alpha, data_color=data_color[idx2], max_models=max_models, color_scheme=color_scheme)
			end
				

			donor_ax = Axis(f[idx, length(populations)], yaxisposition = :right, yticksvisible=false, aspect=1.5)
			hidexdecorations!(donor_ax)
			hideydecorations!(donor_ax, label = false)
			hidespines!(donor_ax)
			donor_ax.ylabel=j
		end
		
		for k in 1:size(ax, 2)
			linkxaxes!(ax[:,k]...)
		end
		
		[hidexdecorations!(j) for j in ax[1:(end-1),:]]
		
		
		[j.ylabel = "enrichment" for j in ax[:, 1]]
		[j.xlabel = "time (days)" for j in ax[end, 1:end]]
		###
	
		data_marker = [MarkerElement(marker = :circle, color = marker_c,
    strokecolor = :transparent, markersize=10) for marker_c in data_color]

		model_color = [PolyElement(color = color, strokecolor = :transparent)
    for color in color_scheme]
		
		cb = f[size(ax,1)+1,:]
		# Legend(cb, ax[1], "prediction")
		if location == "b"
			Legend(cb, [model_color, data_marker],["model " .* string.(models),populations], ["prediction","data"], orientation=:horizontal, titlealign=:left)
		else
			Legend(cb, model_color, "model " .* string.(models), "prediction", position=:top, orientation=:horizontal, titlealign=:left)
		end
		
		return f, ax
	end
end

# ╔═╡ 76a3b35a-8c27-43d3-8148-d8af071bb338
ppc_1 = plot_predictions(df;donors_plotted = ["C66", "C67", "C68"], populations=["ASDC", "cDC1", "cDC2"], dataset="original", models=[1,2,3,4], location ="b", f_kwargs =(;resolution = (800, 600)))[1]

# ╔═╡ 7810af6d-d557-42b3-b585-7e80f57747d6
save(joinpath(res_folder, "ppc_ASDC_original_pooled.pdf"), ppc_1)

# ╔═╡ 5eef1a9e-10fe-4e89-82d5-8df105e7f911
ppc_2 = plot_predictions(df_nonpooled;donors_plotted = ["C66", "C67", "C68"], populations=["ASDC", "cDC1", "cDC2"], dataset="original", models=[1,2,3,4], location ="b", f_kwargs =(;resolution = (800, 600)))[1]

# ╔═╡ 95ffc485-1cde-4429-9e4e-9d0062ac57ee
save(joinpath(res_folder, "ppc_ASDC_original_nonpooled.pdf"), ppc_2)

# ╔═╡ fc1777c6-ad4e-4a43-938a-729c5cb84f18
ppc_3 = plot_predictions(df;donors_plotted = ["C66", "C67", "C68", "C53", "C55"], populations=["ASDC", "cDC1", "cDC2"], dataset="extended", models=[1,2,3,4], location ="b", f_kwargs =(;resolution = (800, 900)))[1]

# ╔═╡ b7583245-ed9c-49e4-b8a7-890f814ac3c5
save(joinpath(res_folder, "ppc_ASDC_extended_pooled.pdf"), ppc_3)

# ╔═╡ 27bec401-50f4-4fed-9a5a-b5acd59902ab
ppc_4 = plot_predictions(df_pdc_pooled;donors_plotted = ["C66", "C67", "C68", "C52"], populations=["pDC"], dataset="original", models=[1,2], location ="b", data_color = [colorant"#c8ab37ff"], max_models=2, f_kwargs =(;resolution = (800, 1000)))[1]

# ╔═╡ eff9e5df-ee87-401e-b97a-1a80a3667477
save(joinpath(res_folder, "ppc_pDC_extended_pooled.pdf"), ppc_4)

# ╔═╡ b79ed428-994c-4683-ac87-18e453e6a261
ppc_5 = plot_predictions(df_pdc_nonpooled;donors_plotted = ["C66", "C67", "C68", "C52"], populations=["pDC"], dataset="original", models=[1,2], location ="b", data_color = [colorant"#c8ab37ff"], max_models=2, f_kwargs =(;resolution = (800, 1000)))[1]

# ╔═╡ c17d27c8-dcce-4c11-82d6-3ba3ad5ea66e
save(joinpath(res_folder, "ppc_pDC_extended_nonpooled.pdf"), ppc_5)

# ╔═╡ 16154f55-7e5e-4741-9ad8-9ec8d7fc391c
md"## Bone marrow prediction"

# ╔═╡ 11f5be0c-2663-4ca0-adf7-a70203161ed7
f = plot_predictions(df;donors_plotted = ["C68"], populations=["ASDC"], dataset="extended", models=[1,2], location ="bm",  max_models=4, alpha = 0.5, f_kwargs =(;resolution = (800, 400)))

# ╔═╡ a5ca7261-198f-4f06-ad1a-fd2870fa0bd4
CairoMakie.xlims!(f[2][1], 0, 4)

# ╔═╡ ddd0ff15-c38d-4a56-ad46-99b027b28b10
f[1]

# ╔═╡ f78b66f8-9601-4ba0-a23a-74fa7e444f43
save(joinpath(res_folder, "ppc_ASDC_extended_pooled_bone_marrow.pdf"), f[1])

# ╔═╡ b8715ebf-5290-4307-a554-7fd0ab3086c6
md"## Model comparison"

# ╔═╡ 715b9bf0-17c5-415c-a1d1-c22d864fd6af
df_loo_pDC = @pipe CSV.read(projectdir("notebooks", "03_analysis",  "JM_0043_Julia_Analysis_pDC","results","PSIS_LOO_CV_Model_comparison_pDC_leave_out_sample.csv"), DataFrame) |>
transform(_, :name => (x -> replace.(x, "_" => " ")), renamecols=false) |>
transform(_, :name => (x -> replace.(x, "Model" => "model")), renamecols=false) |>
transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)

# ╔═╡ 15716182-57e9-4f7a-b685-af321fdb8d8a
df_loo_subset_extended = @pipe CSV.read(projectdir("notebooks", "03_analysis",  "JM_0042_Julia_Analysis_ASDC_cDC1_cDC2","results","PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.csv"), DataFrame) |>
transform(_, :name => (x -> replace.(x, "_" => " ")), renamecols=false) |>
transform(_, :name => (x -> replace.(x, "Model" => "model")), renamecols=false) |>
transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)

# ╔═╡ 25c18a6a-3e7b-4eab-b38c-b6f6156d65e1
df_loo_sample_extended = @pipe CSV.read(projectdir("notebooks", "03_analysis",  "JM_0042_Julia_Analysis_ASDC_cDC1_cDC2","results","PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.csv"), DataFrame) |>
transform(_, :name => (x -> replace.(x, "_" => " ")), renamecols=false) |>
transform(_, :name => (x -> replace.(x, "Model" => "model")), renamecols=false) |>
transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)

# ╔═╡ 56d68aaa-bf87-460d-beb5-420fd7f60fc3
df_loo_subset = @pipe CSV.read(projectdir("notebooks", "03_analysis",  "JM_0042_Julia_Analysis_ASDC_cDC1_cDC2","results","PSIS_LOO_CV_Model_comparison_leave_out_subset.csv"), DataFrame) |>
transform(_, :name => (x -> replace.(x, "_" => " ")), renamecols=false) |>
transform(_, :name => (x -> replace.(x, "Model" => "model")), renamecols=false) |>
transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)

# ╔═╡ ab3577de-6f11-4e9f-b171-f38948b0de09
df_loo_sample = @pipe CSV.read(projectdir("notebooks", "03_analysis",  "JM_0042_Julia_Analysis_ASDC_cDC1_cDC2","results","PSIS_LOO_CV_Model_comparison_leave_out_sample.csv"), DataFrame) |>
transform(_, :name => (x -> replace.(x, "_" => " ")), renamecols=false) |>
transform(_, :name => (x -> replace.(x, "Model" => "model")), renamecols=false) |>
transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)

# ╔═╡ dddde160-19ee-4a91-b22e-e267cbda3ce7


# ╔═╡ eb5f85bd-5324-4e73-91fe-efc42d630176
begin
	function plot_model_comparison(df; resolution=(600,400), f = CairoMakie.Figure(; resolution), ax = Axis(f[1,1]), colors=:roma, max_models=5)
		
		
		order_model = sortperm(sortperm(string.(df.name)))
		
		# color_scheme= get(colorschemes[color], LinRange(0,1,nrow(df)))
		color_scheme = cgrad(colors,max_models, categorical=true)[order_model]
		CairoMakie.errorbars!(ax,df.loo,  collect(1:nrow(df)), df.se, direction=:x, color=:black,linewidth=2, whiskerwidth=8)

		@pipe data(df) * mapping(:loo, :name, color=:name, group=:name) * visual(Scatter, markersize=10) |> draw!(ax, _; palettes=(color=color_scheme,))

		CairoMakie.errorbars!(ax,df.loo[2:end],  collect(2:nrow(df)).+ 0.1, df.dse[2:end], direction=:x, whiskerwidth=8, linewidth=2, color=:grey)

		CairoMakie.scatter!(ax, df.loo[2:end], collect(2:nrow(df)) .+ 0.1, color=:grey, markersize=8)

		CairoMakie.vlines!(ax, maximum(df.loo), color=:grey, linestyle = :dash, linewidth=1)
		ax.ylabel =""
		ax.xlabel ="elpd (greater is better)"

		return f
	end
end

# ╔═╡ ae35a1df-d594-4de8-946c-de827309cece
begin
	fig_pdc_model_comparison = CairoMakie.Figure(resolution = (400, 350))
	ax_pdc_model_comparison = Axis(fig_pdc_model_comparison[1,1], title="sample")
	
	plot_model_comparison(df_loo_pDC; ax = ax_pdc_model_comparison)
	fig_pdc_model_comparison
end

# ╔═╡ d811ceb6-b60f-420a-962c-8d6a121588d1
save(joinpath(res_folder, "model_comparison_pDC_extended_pooled.pdf"), fig_pdc_model_comparison)

# ╔═╡ 444f287e-883f-4f43-a6fd-5c01474d635d
plot_model_comparison(df_loo_sample)

# ╔═╡ 209350c1-6cce-452a-87fc-68324672d914
plot_model_comparison(df_loo_subset)

# ╔═╡ cc028ef7-21e5-4666-8ea3-3463c8fb4c4b
plot_model_comparison(df_loo_sample_extended)

# ╔═╡ da17f29a-baf8-4cfb-9aab-f27eac119788
plot_model_comparison(df_loo_subset_extended)

# ╔═╡ a5ce5ea7-8622-4a40-a459-e0728ae162e3
noto_sans_bold = assetpath("fonts", "NotoSans-Bold.ttf")

# ╔═╡ 9fc567d3-5519-4ec1-b749-ec36ceea856b
begin
	fig_pdc_ppc_combined = CairoMakie.Figure(; resolution = (800,1000))
	ax1 = hcat([[Axis(fig_pdc_ppc_combined[k,j]) for k in 1:4] for j in 1:1]...)
	ax2 = hcat([[Axis(fig_pdc_ppc_combined[k,j]) for k in 1:4] for j in 2:2]...)
	plot_predictions(df_pdc_pooled;donors_plotted = ["C66", "C67", "C68", "C52"], populations=["pDC"], dataset="original", models=[1,2], location ="b", data_color = [colorant"#c8ab37ff"], max_models=2, ax = ax1)[2]
plot_predictions(df_pdc_nonpooled;donors_plotted = ["C66", "C67", "C68", "C52"], populations=["pDC"], dataset="original", models=[1,2], location ="b", data_color = [colorant"#c8ab37ff"], max_models=2, ax=ax2)[2]

	donor_ax = [Axis(fig_pdc_ppc_combined[j, 2], yaxisposition = :right, yticksvisible=false, aspect=1.5) for j in 1:4]
	[hidexdecorations!(j) for j in donor_ax]
	[hideydecorations!(j, label = false) for j in donor_ax]
	[hidespines!(j) for j in donor_ax]
	
	[donor_ax[j].ylabel=["C66", "C67", "C68", "C52"][j] for j in 1:4]
	[hideydecorations!(j, ticklabels=false, ticks=false) for j in ax2]
	[linkyaxes!(ax1[j], ax2[j]) for j in 1:4]
	
	data_marker = [MarkerElement(marker = :circle, color = marker_c, strokecolor = :transparent, markersize=10) for marker_c in [colorant"#c8ab37ff"]]

	model_color = [PolyElement(color = color, strokecolor = :transparent) for color in cgrad(:roma, 2, categorical =true, alpha=0.5)]
		
	cb = fig_pdc_ppc_combined[4+1,:]
	# Legend(cb, ax[1], "prediction")
	Legend(cb, [model_color, data_marker],["model " .* string.([1,2]),["pDC"]], ["prediction","data"], orientation=:horizontal, titlealign=:left)
	
	ax1[1].title = "pDC - pooled model"
	ax2[1].title = "pDC - nonpooled model"

	for (label, layout) in zip(["A", "B"], [fig_pdc_ppc_combined[1,1], fig_pdc_ppc_combined[1,2]])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 26,
        font = noto_sans_bold,
        padding = (0, label == "A" ? 45 : 30, 5, 0),
        halign = :left)
	end
	# Legend(fig_pdc_ppc_combined[5,:], ax1[1], orientation=:horizontal)
	# fig_pdc_ppc_combined.resolution= (100,100)
	fig_pdc_ppc_combined
end

# ╔═╡ 2f790fe7-6b7d-4b7a-8b60-89b416b790e1
save(joinpath(res_folder, "ppc_pDC_extended_combined_nonpooled_pooled.pdf"), fig_pdc_ppc_combined)

# ╔═╡ d6fa1146-712d-4e2d-a69c-aaadfaf480fa
noto_sans = assetpath("fonts", "NotoSans-Regular.ttf")

# ╔═╡ 7882d78f-269a-4c44-844c-2dedf18a5199
begin
	fig_ASDC_comparison = CairoMakie.Figure(; resolution = (800,400), font = noto_sans)
	gp_sample = fig_ASDC_comparison[1,1]
	gp_subset = fig_ASDC_comparison[1,2]
	ax_sample = Axis(gp_sample, title="Sample", aspect=1)
	ax_subset = Axis(gp_subset, title="Subset", aspect=1)
	
	
	plot_model_comparison(df_loo_sample; ax=ax_sample)
	plot_model_comparison(df_loo_subset; ax=ax_subset)
	
	
	for (label, layout) in zip(["A", "B"], [gp_sample, gp_subset])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 26,
        font = noto_sans_bold,
        padding = (0, 48, 5, 0),
        halign = :left)
	end
	
	fig_ASDC_comparison
	
end

# ╔═╡ 643a399d-b2bd-4bed-afa5-6e6ca4e13dd1
save(joinpath(res_folder, "model_comparison_ASDC_original_pooled.pdf"), fig_ASDC_comparison)

# ╔═╡ 04313499-f599-4f73-9c30-a0b214448090
begin
	fig_ASDC_comparison_extended = CairoMakie.Figure(; resolution = (800,400),font = noto_sans)
	gp_sample_extended = fig_ASDC_comparison_extended[1,1]
	gp_subset_extended = fig_ASDC_comparison_extended[1,2]
	ax_sample_extended = Axis(gp_sample_extended, title="Sample", aspect=1)
	ax_subset_extended = Axis(gp_subset_extended, title="Subset", aspect=1)
	
	
	plot_model_comparison(df_loo_sample_extended; ax=ax_sample_extended)
	plot_model_comparison(df_loo_subset_extended; ax=ax_subset_extended)
	
	
	for (label, layout) in zip(["A", "B"], [gp_sample_extended, gp_subset_extended])
    Label(layout[1, 1, TopLeft()], label,
        textsize = 26,
        font = noto_sans_bold,
        padding = (0, 48, 5, 0),
        halign = :left)
	end
	
	fig_ASDC_comparison_extended
	
end

# ╔═╡ 930678b5-882d-4042-b087-09bb7561c8f8
save(joinpath(res_folder, "model_comparison_ASDC_extended_pooled.pdf"), fig_ASDC_comparison_extended)

# ╔═╡ 3cf77b33-2d19-490f-a863-ca376d8c329e
md" ## Marginal posteriors"

# ╔═╡ bd84e89e-8091-4613-a7d5-f5dd56486819
md"Load posterior dataframes"

# ╔═╡ 5e59b133-ddae-42b2-86a4-0d880a739b07
begin
	df_full_posterior_extended = DataFrame()
	for j in [1,2,3,4]
		global df_full_posterior_extended = @pipe CSV.read(projectdir("notebooks", "03_analysis", "JM_0042_Julia_Analysis_ASDC_cDC1_cDC2","results","Parameter_full_posterior_model_$(j).csv"), DataFrame) |> vcat(df_full_posterior_extended,_, cols=:union)
	end
	df_full_posterior_extended = @pipe df_full_posterior_extended |> rename(_, :model_id => :model)
end

# ╔═╡ 1f080edd-b6ec-4350-978c-2cd19d5b5860
md" corner plot of all parameters"

# ╔═╡ 4dc94cfe-f6d1-4310-a444-9f6edfeb0c91
function plot_posterior_distribution(df, parameters, models; xlabel=L"d^{-1}",  resolution=(600,400), f = CairoMakie.Figure(; resolution), sf = f[1,1], colors=:roma, alpha=0.5, offset_factor = 2, lims_factor=4, max_models=4)
	
	color_scheme = cgrad(colors, max_models, categorical = true, alpha=alpha)[models]
	
	@pipe df |>
	transform(_, :model => (x -> categorical(x, compress=true)), renamecols=false) |>
	subset(_, :model => (x -> x .∈ Ref(models))) |>
	select(_, Not([:model_type, :donor])) |> 
	select(_, :model, parameters) |>
	DataFrames.stack(_, Not(:model)) |>
	rename(_,:variable => :parameter) |>
	groupby(_, :parameter) |>
	begin
		tmp_ax = [Axis(sf[1,j], title=first(keys(_)[j]), yticks=((1:length(models)) ./ offset_factor, "model " .* string.(models)), xlabel=xlabel, aspect=1) for j in 1:length(keys(_))]
		for (idx, j) in enumerate(keys(_))
			tmp_df1 = _[NamedTuple(j)]
			tmp_gdf1 = groupby(DataFrame(tmp_df1), :model)
			for (idx2, k) in enumerate(models)
				tmp_df2 = tmp_gdf1[(; model=k)]
				model_tmp = map(x->tryparse.(Int, string(x)), tmp_df2.model)
				value_tmp = Array{Float64, 1}(tmp_df2.value)
				ci = MCMCChains._hpd(value_tmp)
				m = mean(value_tmp)

				d = CairoMakie.density!(tmp_ax[idx], Array{Float64, 1}(tmp_df2.value), color =color_scheme[idx2], offset = idx2/offset_factor,strokewidth = 1, strokecolor = :black)
				s = CairoMakie.scatter!(tmp_ax[idx], [m], [idx2/offset_factor], color=:black)
				e  = CairoMakie.rangebars!(tmp_ax[idx], [idx2/offset_factor], [ci[1]], [ci[2]], color=:black, direction =:x)
				
				
				
				current_lims = tmp_ax[idx].limits[][1]
				
				min_post = m - lims_factor* std(value_tmp)
				max_post = m + lims_factor* std(value_tmp)
				
				if !isnothing(current_lims)
					current_xlim_l = tmp_ax[idx].limits[][1][1]
					current_xlim_u = tmp_ax[idx].limits[][1][2]

					update_xlim_l = maximum([minimum([min_post,current_xlim_l]), -0.1, ])
					update_xlim_u = maximum([max_post, current_xlim_u])
				else
					update_xlim_l = maximum([min_post, -0.1])
					update_xlim_u = max_post
				end

				CairoMakie.xlims!(tmp_ax[idx],update_xlim_l, update_xlim_u)
				
				CairoMakie.translate!(d, 0, 0, -0.1idx2)
				CairoMakie.translate!(s, 0, 0, -0.1idx2)

				CairoMakie.translate!(e, 0, 0, -0.1idx2)

			end
		end
		CairoMakie.linkyaxes!(tmp_ax...)
		[CairoMakie.hideydecorations!(j) for j in tmp_ax[2:end]]
		# [j.xlabel=xlabel for j in tmp_ax[1:end]]
		#color_scheme[model_tmp]
	end
# 	data(_) * mapping(:model, :value, col=:parameter, color=:model) *visual(Violin, show_median=true)|>
# 	AlgebraOfGraphics.draw!(sf, _; facet=(; linkyaxes=:minimal, linkxaxes=:none), palettes=(color=cgrad(:roma, maximum(models), categorical=true)[models],))
	
	return f, sf, tmp_ax
	
end

# ╔═╡ 55880998-8061-4f3a-b751-116e4901b10b
begin
	fig_dwell = CairoMakie.Figure(resolution=(800,600))
	sf_dwell1 = fig_dwell[1,1]
	sf_dwell2 = fig_dwell[2,1]


plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> startswith(x, "dwell_"), DataFrames.names(df_full_posterior_extended)))[collect(1:3)], [1,2,4]; sf=sf_dwell1, alpha=0.5, offset_factor=0.5, xlabel="")
	plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> startswith(x, "dwell_"), DataFrames.names(df_full_posterior_extended)))[collect(4:6)], [1,2,4]; sf=sf_dwell2, alpha=0.5, offset_factor=0.5, xlabel="days")
	
	fig_dwell
end

# ╔═╡ 27dc81d6-e986-4a5b-9b56-81e47def9080
save(joinpath(res_folder, "marginal_posteriors_dwell_ASDC_extended_pooled.pdf"), fig_dwell)

# ╔═╡ 122a5792-8c9a-4200-b79b-8d32e2f3c3c7
begin
	
	fig_prol = CairoMakie.Figure(resolution=(800,400))
	sf_prol = fig_prol[1,1]

plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> startswith(x, "p_"), DataFrames.names(df_full_posterior_extended)))[collect(1:3)], [1,2,4]; sf=sf_prol, alpha=0.5, offset_factor=0.5, xlabel="day⁻¹")
	
	fig_prol
end

# ╔═╡ 36b84ebc-1b5a-496b-8cd2-2e94d7201efd
save(joinpath(res_folder, "marginal_posteriors_proliferation_ASDC_extended_pooled.pdf"), fig_prol)

# ╔═╡ 62674e98-3a3a-48b7-860a-012cac18de8a
begin
	
	fig_trans = CairoMakie.Figure(resolution=(800,400))
	sf_trans = fig_trans[1,1]

plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> startswith(x, "λ_"), DataFrames.names(df_full_posterior_extended))), [1,2,4]; sf=sf_trans, alpha=0.5, offset_factor=0.5, xlabel="day⁻¹")
	
	fig_trans
end

# ╔═╡ 8b71b338-63ad-458b-bc90-6abe3247fddd
save(joinpath(res_folder, "marginal_posteriors_transition_ASDC_extended_pooled.pdf"), fig_trans)

# ╔═╡ ce20213f-6a62-494d-82f7-1bfdb83b5191
begin
	fig_death = CairoMakie.Figure(resolution=(800,600))
	sf_death1 = fig_death[1,1]
	sf_death2 = fig_death[2,1]


plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> startswith(x, "δ_"), DataFrames.names(df_full_posterior_extended)))[collect(1:3)], [1,2,4]; sf=sf_death1, alpha=0.5, offset_factor=0.2, xlabel="")
	plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> startswith(x, "δ_"), DataFrames.names(df_full_posterior_extended)))[collect(4:6)], [1,2,4]; sf=sf_death2, alpha=0.5, offset_factor=0.5, xlabel="day⁻¹")
	
	fig_death
end

# ╔═╡ 22eb6659-5ea9-4e83-b3fc-6e331e64ddff
save(joinpath(res_folder, "marginal_posteriors_death_ASDC_extended_pooled.pdf"), fig_death)

# ╔═╡ 20411334-d5ae-49f2-abf7-8c03ea4b8d4c
begin
	fig_diff = CairoMakie.Figure(resolution=(600,600))
	sf_diff1 = fig_diff[1,1]
	sf_diff2 = fig_diff[1,2]
	sf_diff3 = fig_diff[2,1]
	sf_diff4 = fig_diff[2,2]
	
	ax_diff1 = plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> endswith(x, "Δ_cDC1bm"), DataFrames.names(df_full_posterior_extended))), [1,2]; sf=sf_diff1, alpha=0.5, offset_factor=1.5, xlabel="")[3]
	ax_diff2 = plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> endswith(x, "Δ_cDC2bm"), DataFrames.names(df_full_posterior_extended))), [1,2,4]; sf=sf_diff2, alpha=0.5, offset_factor=1.5,xlabel="")[3]
	ax_diff3 = plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> endswith(x, "Δ_cDC1b"), DataFrames.names(df_full_posterior_extended))), [1]; sf=sf_diff3, alpha=0.5, offset_factor=1.5, xlabel="day⁻¹")[3]
	ax_diff4 = plot_posterior_distribution(df_full_posterior_extended, Symbol.(filter(x -> endswith(x, "Δ_cDC2b"), DataFrames.names(df_full_posterior_extended))), [1,4]; sf=sf_diff4, alpha=0.5, offset_factor=1.5, xlabel="day⁻¹")[3]
	
	linkyaxes!(ax_diff1[1], ax_diff2[1])
	linkyaxes!(ax_diff3[1], ax_diff4[1])

	# [hidexdecorations!(j) for j in [ax_diff1[1], ax_diff2[1]]]
	
	fig_diff
end

# ╔═╡ e74ea368-ffe4-44cf-92e6-12cd5132986a
save(joinpath(res_folder, "marginal_posteriors_differentiation_ASDC_extended_pooled.pdf"), fig_diff)

# ╔═╡ daf359d5-cd50-48a1-9b79-7e8fe0c3c94d
md"Posterior plots of proliferation rate and dwell time in more detail"

# ╔═╡ 968d82e2-8307-41e9-880e-409c3c405add
md"## pDC estimates"

# ╔═╡ 4d70cb69-5e66-4f72-8fc5-ac8f60abc055
begin
	df_full_pDC_posterior_extended = DataFrame()
	for j in [1,2]
		global df_full_pDC_posterior_extended = @pipe CSV.read(projectdir("notebooks", "03_analysis", "JM_0043_Julia_Analysis_pDC","results","Parameter_full_posterior_pDC_model_$(j).csv"), DataFrame) |> vcat(df_full_pDC_posterior_extended,_, cols=:union)
	end
	df_full_pDC_posterior_extended = @pipe df_full_pDC_posterior_extended |> rename(_, :model_id => :model)
end

# ╔═╡ e2a1684e-0771-4d17-ba08-c14389f14781
begin
	
	fig_trans_pdc = CairoMakie.Figure(resolution=(600,400))
	sf_trans_pdc1 = fig_trans_pdc[1,1]
	sf_trans_pdc2 = fig_trans_pdc[1,2]

	
plot_posterior_distribution(df_full_pDC_posterior_extended, [:λ_pDC], [1,2]; sf=sf_trans_pdc1, alpha=0.5, offset_factor=0.3, xlabel="day⁻¹")
	
	plot_posterior_distribution(df_full_pDC_posterior_extended, [:tau], [2]; sf=sf_trans_pdc2, alpha=0.5, offset_factor=0.3, xlabel="days")
	
	fig_trans_pdc
end

# ╔═╡ d124ac4f-bb0e-4a22-aa15-985d91ebc7c4
save(joinpath(res_folder, "marginal_posteriors_transition_pDC_extended_pooled.pdf"), fig_trans_pdc)

# ╔═╡ 090bc205-3be5-4c2c-a8a1-fcc7729ef8b6
begin
	
	fig_tau_pdc = CairoMakie.Figure(resolution=(600,400))
	sf_tau_pdc = fig_tau_pdc[1,1]

	plot_posterior_distribution(df_full_pDC_posterior_extended, [:tau], [2]; sf=sf_tau_pdc, alpha=0.5, offset_factor=0.3, xlabel="days")
	
	fig_tau_pdc
end

# ╔═╡ 071a38aa-87c8-4ac5-bf42-de61d8de3806
begin
	
	fig_prol_pdc = CairoMakie.Figure(resolution=(600,400))
	sf_prol_pdc = fig_prol_pdc[1,1]

plot_posterior_distribution(df_full_pDC_posterior_extended, Symbol.(filter(x -> startswith(x, "p_"), DataFrames.names(df_full_pDC_posterior_extended))), [1,2]; sf=sf_prol_pdc, alpha=0.5, offset_factor=0.3, lims_factor=6, xlabel="day⁻¹")
	
	fig_prol_pdc
end

# ╔═╡ 5eed1036-5316-4ce6-a65f-000e80e04087
save(joinpath(res_folder, "marginal_posteriors_proliferation_pDC_extended_pooled.pdf"), fig_prol_pdc)

# ╔═╡ df146e27-5d96-4300-bac7-2e496aeea834
begin
	
	fig_death_pdc = CairoMakie.Figure(resolution=(600,400))
	sf_death_pdc = fig_death_pdc[1,1]

plot_posterior_distribution(df_full_pDC_posterior_extended, Symbol.(filter(x -> startswith(x, "δ_"), DataFrames.names(df_full_pDC_posterior_extended))), [1,2]; sf=sf_death_pdc, alpha=0.5, offset_factor=0.3, lims_factor=5, xlabel="day⁻¹")
	
	fig_death_pdc
end

# ╔═╡ 5e07ffe2-5ce1-4b1e-94bd-624ec13753dd
save(joinpath(res_folder, "marginal_posteriors_death_pDC_extended_pooled.pdf"), fig_death_pdc)

# ╔═╡ 715e8518-bb57-42a7-936d-f4437953c887
begin
	
	fig_dwell_pdc = CairoMakie.Figure(resolution=(600,400))
	sf_dwell_pdc = fig_dwell_pdc[1,1]

plot_posterior_distribution(df_full_pDC_posterior_extended, Symbol.(filter(x -> startswith(x, "dwell_"), DataFrames.names(df_full_pDC_posterior_extended))), [1,2]; sf=sf_dwell_pdc, alpha=0.5, offset_factor=4, lims_factor=5, xlabel="day⁻¹")
	
	fig_dwell_pdc
end

# ╔═╡ 41e3d9f5-32a4-4942-be5a-ae66c2180612
save(joinpath(res_folder, "marginal_posteriors_dwell_pDC_extended_pooled.pdf"), fig_dwell_pdc)

# ╔═╡ ef43158e-2659-4657-9164-c59ab9735e2e
# @pipe df_full_pDC_posterior_extended |>
# subset(_, :model => x -> x .== 2) |>
# select(_, Not([:model, :model_type, :donor, :prior])) |>
# transform(_, :tau => (x -> Array{Float64,1}(x)), renamecols=false) |>
# DataFrames.stack(_) |>
# groupby(_, :variable) |>
# combine(_, :value => (x -> (mean=mean(x), (; zip((:lower, :upper), MCMCChains._hpd(Array(x)))...)...))=> AsTable) |>
# rename(_, :variable => :parameter) |>
# DataFrames.transform(_, DataFrames.names(_, Float64) .=> (x -> round.(x, digits=4)), renamecols=false) |>
# df2latex(_, joinpath(res_folder, "tab_pDC_posterior_model_2_summary.tex"))

# ╔═╡ b2907b71-fde3-4e90-8b9e-8c0774db3cb7
res_folder

# ╔═╡ 9264d2c2-ce27-40b9-8c44-5ca25f949641
md"## Divergent transitions and k shape parameters"

# ╔═╡ 96c5f88c-48a6-4e2e-94f4-ee42d4cd9701
md"Check out if old normal model inference have most of the information we are interested in"

# ╔═╡ 64d3324d-c2e8-4255-a158-d315d14664c0
md"load mcmc_res.jlso from notebook folders to numerate divergent transitions"

# ╔═╡ 1e365da6-9b66-4d5e-9897-e652bf988a76
begin
	df_divergence_pooled = DataFrame()
	for (idx, j) in enumerate(pooled_results_notebooks)
		df_tmp = DataFrame(:n_divergence => sum(get(JLSO.load(joinpath(j,"results", "mcmc_res.jlso"))[:chain], :numerical_error)[1]), :data => data_input_pooled[idx], :strata => strata_pooled[idx], :model => model_id_pooled[idx], :prior=>priors_pooled[idx], :likelihood_f => likelihood_pooled[idx])
		
		df_divergence_pooled = vcat(df_divergence_pooled, df_tmp, cols=:union)
	end
	subset!(df_divergence_pooled, :model => x -> x .∈ Ref([1,2,4,5]))
	transform!(df_divergence_pooled, :model => (x -> replace(x, 4 => 3, 5 => 4)), renamecols=false) 
end

# ╔═╡ b4e37055-8fe4-4c7d-b790-eff1fcc242a6
begin
	df_divergence_normal_pooled = DataFrame()
	for (idx, j) in enumerate(pooled_results_normal_notebooks)
		df_tmp = DataFrame(:n_divergence => sum(get(JLSO.load(joinpath(j,"results", "mcmc_res.jlso"))[:chain], :numerical_error)[1]), :data => data_input_normal_pooled[idx], :strata => strata_normal_pooled[idx], :model => model_id_normal_pooled[idx], :prior=>priors_normal_pooled[idx], :likelihood_f => likelihood_normal_pooled[idx])
		
		df_divergence_normal_pooled = vcat(df_divergence_normal_pooled, df_tmp, cols=:union)
	end
	df_divergence_normal_pooled
		subset!(df_divergence_normal_pooled, :model => x -> x .∈ Ref([1,2,4,5]))
	transform!(df_divergence_normal_pooled, :model => (x -> replace(x, 4 => 3, 5 => 4)), renamecols=false) 
end

# ╔═╡ ad92cff6-b519-44f1-9a8d-699b6ec191b8
begin
	df_divergence_nonpooled = DataFrame()
	for (idx, j) in enumerate(nonpooled_results_notebooks)
		df_tmp = DataFrame(:n_divergence => sum(get(JLSO.load(joinpath(j,"results", "mcmc_res.jlso"))[:chain], :numerical_error)[1]), :data => data_input_nonpooled[idx], :strata => strata_nonpooled[idx], :model => model_id_nonpooled[idx], :prior=>priors_nonpooled[idx], :likelihood_f => likelihood_nonpooled[idx])
		
		df_divergence_nonpooled = vcat(df_divergence_nonpooled, df_tmp, cols=:union)
	end
	df_divergence_nonpooled
			subset!(df_divergence_nonpooled, :model => x -> x .∈ Ref([1,2,4,5]))
	transform!(df_divergence_nonpooled, :model => (x -> replace(x, 4 => 3, 5 => 4)), renamecols=false) 
end

# ╔═╡ 09b1fdc7-4803-4922-9537-d6ca2a91d630
begin
	df_divergence_normal_nonpooled = DataFrame()
	for (idx, j) in enumerate(nonpooled_results_normal_notebooks)
		if isfile(joinpath(j,"results", "mcmc_res.jlso"))
			df_tmp = DataFrame(:n_divergence => sum(get(JLSO.load(joinpath(j,"results", "mcmc_res.jlso"))[:chain], :numerical_error)[1]), :data => data_input_normal_nonpooled[idx], :strata => strata_normal_nonpooled[idx], :model => model_id_normal_nonpooled[idx], :prior=>priors_normal_nonpooled[idx], :likelihood_f => likelihood_normal_nonpooled[idx])
		
			df_divergence_normal_nonpooled = vcat(df_divergence_normal_nonpooled, df_tmp, cols=:union)
		end
	end
	df_divergence_normal_nonpooled
	
				subset!(df_divergence_normal_nonpooled, :model => x -> x .∈ Ref([1,2,4,5]))
	transform!(df_divergence_normal_nonpooled, :model => (x -> replace(x, 4 => 3, 5 => 4)), renamecols=false) 
end

# ╔═╡ ffbd64f9-1d69-4156-b609-3045db464ecc
begin
	df_divergence_pooled_pdc = DataFrame()
	for (idx, j) in enumerate(pooled_pdc_results_notebooks)
		df_tmp = DataFrame(:n_divergence => sum(get(JLSO.load(joinpath(j,"results", "mcmc_res.jlso"))[:chain], :numerical_error)[1]), :data => data_input_pdc_pooled[idx], :strata => strata_pdc_pooled[idx], :model => model_id_pdc_pooled[idx], :prior=>priors_pdc_pooled[idx], :likelihood_f => likelihood_pdc_pooled[idx])
		
		df_divergence_pooled_pdc = vcat(df_divergence_pooled_pdc, df_tmp, cols=:union)
	end
	df_divergence_pooled_pdc
end

# ╔═╡ 76ec3dc3-7f7b-4af7-8569-02458f6df245
begin
	df_divergence_nonpooled_pdc = DataFrame()
	for (idx, j) in enumerate(nonpooled_pdc_results_notebooks)
		df_tmp = DataFrame(:n_divergence => sum(get(JLSO.load(joinpath(j,"results", "mcmc_res.jlso"))[:chain], :numerical_error)[1]), :data => data_input_pdc_nonpooled[idx], :strata => strata_pdc_nonpooled[idx], :model => model_id_pdc_nonpooled[idx], :prior=>priors_pdc_nonpooled[idx], :likelihood_f => likelihood_pdc_nonpooled[idx])
		
		df_divergence_nonpooled_pdc = vcat(df_divergence_nonpooled_pdc, df_tmp, cols=:union)
	end
	df_divergence_nonpooled_pdc
end

# ╔═╡ 1af67a87-7033-4a83-aa22-860b199bd4f0
begin
	@pipe df_divergence_normal_nonpooled |> subset(_, :prior => x -> x .== "lognormal")|> data(_)*mapping(:model, :n_divergence, color=:model => (x -> string.(x))) * visual(BarPlot) |> AlgebraOfGraphics.draw(_; palettes=(color=cgrad(:roma, 4, categorical=true),))
end

# ╔═╡ 94b38a9b-2e78-4b56-b55a-15c5424b8267
md"Load loglikelihood materices"

# ╔═╡ d8c18079-d2d5-4d31-845a-6a176f2084f0
begin
	arr_pointwise_loglike_pooled_normal = Array[]
	
	for j in pooled_results_normal_notebooks
		tmp_filename = joinpath(j,"results", "logp_3d_mat.jlso")
		if isfile(tmp_filename)
			tmp_mat = JLSO.load(tmp_filename)[:loglike_3d]	
			push!(arr_pointwise_loglike_pooled_normal, tmp_mat)
		end
	end
	
	arr_pointwise_loglike_pooled_normal
	
end

# ╔═╡ f5104a54-fc66-464e-9716-1e266c266133
begin
	arr_pointwise_loglike_nonpooled_normal = Array[]
	
	for j in nonpooled_results_normal_notebooks
		tmp_filename = joinpath(j,"results", "logp_3d_mat.jlso")
		if isfile(tmp_filename)
			tmp_mat = JLSO.load(tmp_filename)[:loglike_3d]	
			push!(arr_pointwise_loglike_nonpooled_normal, tmp_mat)
		end
	end
	
	arr_pointwise_loglike_nonpooled_normal
	
end

# ╔═╡ 3436a15e-d49e-436c-aeb8-d7937d11ffeb
begin
	mat_logl=permutedims(arr_pointwise_loglike_pooled_normal[1], [2,3,1])
	@rput mat_logl
	R"r_eff <- relative_eff(exp(mat_logl))"
	R"r_loo_res = loo(mat_logl)"
	@rget r_loo_res
end

# ╔═╡ 271ed04f-3e14-43ba-969f-bdb63c054c09
pareto_res = ParetoSmooth.psis_loo(arr_pointwise_loglike_pooled_normal[1])

# ╔═╡ b1df0dc5-7258-4846-a027-02b215adf976
begin
	@pipe pareto_res.pointwise |> DataFrame(_) |> subset(_, :statistic => x -> x .== :pareto_k) |> subset(_, :value => x -> x .>= 0.7)
	
end

# ╔═╡ 5f262735-7e42-40ef-ae9d-bb7c90f6dcf8
filter(x -> x .>= 0.7, r_loo_res[:diagnostics][:pareto_k])

# ╔═╡ 6e401f0d-5935-4d6f-bc96-4c40f8b08d9f


# ╔═╡ 3d13a62a-51ff-4cf1-bf9e-8228a065088e
arr_inference_data_pooled_normal = [ArviZ.from_mcmcchains(JLSO.load(joinpath(pooled_results_normal_notebooks[j],"results", "mcmc_res.jlso"))[:chain], log_likelihood = permutedims(arr_pointwise_loglike_pooled_normal[j], [3,2,1])) for j in 1:10]

# ╔═╡ 5c4c39db-8735-40e8-94f9-7731c2baa045
arr_inference_data_nonpooled_normal = [ArviZ.from_mcmcchains(JLSO.load(joinpath(nonpooled_results_normal_notebooks[j],"results", "mcmc_res.jlso"))[:chain], log_likelihood = permutedims(arr_pointwise_loglike_nonpooled_normal[j], [3,2,1])) for j in 1:5]

# ╔═╡ 4c575172-48e3-4fb2-b373-67c04e5590d8
begin
	arr_loo_res_pooled = [ArviZ.loo(j, pointwise=true) for j in arr_inference_data_pooled_normal]
	k_res_pooled = [[convert(Float64, k) for k in j.pareto_k[1]] for j in arr_loo_res_pooled]
end

# ╔═╡ b8fb29c6-27c2-4551-93fd-a65534519b0f
begin
	arr_loo_res_nonpooled = [ArviZ.loo(j, pointwise=true) for j in arr_inference_data_nonpooled_normal]
	k_res_nonpooled = [[convert(Float64, k) for k in j.pareto_k[1]] for j in arr_loo_res_nonpooled]
end

# ╔═╡ 6eb38d27-aa50-4309-8f96-7d601cc0277b
begin
# 	model_idx_plot = collect(1:10)
# 	model_name_plot = repeat(collect(1:5), outer=2)
# 	model_strata_plot =
	
	fig_shape_parameter = CairoMakie.Figure(; resolution=(800,500))
	ax_fig_shape1 = Axis(fig_shape_parameter[1,1], xlabel="datapoint index", ylabel="shape parameter k", title="Model 1 - pooled", aspect=1)
	ax_fig_shape2 = Axis(fig_shape_parameter[1,2], xlabel="datapoint index", ylabel="shape parameter k", title="Model 1 - nonpooled", aspect=1)
	
	
	k_res_plot1 = k_res_pooled[1]
	k_res_plot2 = k_res_nonpooled[1]

	
	CairoMakie.scatter!(ax_fig_shape1, collect(1:length(k_res_plot1)), k_res_plot1, marker=:cross, color=cgrad(:roma,2,categorical =true)[1], markersize=15)
	CairoMakie.hlines!(ax_fig_shape1, 0.7, linestyle=:dash, color=:grey)
	
		CairoMakie.scatter!(ax_fig_shape2, collect(1:length(k_res_plot2)), k_res_plot2, marker=:cross, color=cgrad(:roma,2,categorical =true)[1], markersize=15)
	CairoMakie.hlines!(ax_fig_shape2, 0.7, linestyle=:dash, color=:grey)

	linkyaxes!(ax_fig_shape1, ax_fig_shape2)
	hideydecorations!(ax_fig_shape2)
	
	fig_shape_parameter
end

# ╔═╡ c058ef32-a609-4ded-85e7-8ada7426c1d3
save(joinpath(res_folder, "Fig_shape_parameter_k_normal_model_1_pooled_v_nonpooled.pdf"), fig_shape_parameter)

# ╔═╡ Cell order:
# ╠═f6a855d5-f852-437b-913b-48a2f1a24469
# ╠═c671ec28-4bc2-11ec-05c2-1952e68a6e4b
# ╠═5240145d-ba36-479c-bda9-4944e13cf70b
# ╠═2ebde24d-634b-467b-8ee0-6d45174f72f3
# ╠═69516fb5-c504-4fb7-a633-f3e8c42190a9
# ╠═8dca14b1-b8db-4eeb-8bf4-af41936e2505
# ╠═f80272ab-7379-4be0-99c8-bd803b0f48ed
# ╠═e8a0497c-1d9b-465d-a609-ad6fa4f52bec
# ╠═0b015757-d6cb-44c6-94f5-419b46c0dc75
# ╠═a08682c5-a805-4a82-94f2-a210d48a87bb
# ╠═ab676ff6-4906-4204-a9f1-f4c7e3c385cc
# ╠═5b55c1d5-5fdf-462e-bdba-192a589cfc03
# ╠═969d39ad-8696-464f-9087-2eb1e886e94e
# ╠═65129028-c29e-4ab4-a154-62a3a7dcdd5d
# ╠═4f04a6a7-73fa-42f2-8bbe-ddfaadbbefbd
# ╠═bb1f6478-82d5-49cd-8225-51281d911e2d
# ╠═268825ca-d8e7-4f1c-ae91-64747a875984
# ╠═e82c870e-8b63-4ddd-9289-2eb7c9e3cd72
# ╠═8691540d-c347-4fff-914b-00cea31dc22c
# ╠═8f25ba44-552a-41a3-b7ec-32fe28702964
# ╠═1c20d417-f83d-4210-a17f-f8556b6e74ca
# ╠═df7db22c-79c2-4c1e-8f20-cfeb75bcf7c6
# ╠═e128f1e6-b3ef-4d48-97b7-c96248bc84f1
# ╠═ab51a99b-0c05-4f16-a7c2-97852d83a1e6
# ╠═d6f0af2f-3449-417e-bf97-5fdbb8646c46
# ╠═8e9b4cfe-25bf-4311-bbb0-e96b183ac8a0
# ╠═caac50f7-8b6a-4940-9d10-9e58a1e31ccb
# ╠═d84a78ea-394c-4151-b5e4-1dff040e634f
# ╠═a123f0fa-b0c9-4bf3-a2a3-166339bc1999
# ╠═676788cc-b8ee-47ba-9b9e-2746155132ab
# ╠═12454b5d-a771-4044-8ef9-ba08ebcb3ff4
# ╠═0e40f982-bc34-491f-bdef-143c6b2008cd
# ╠═251e83f8-fb1e-431d-ae95-1d3639a5db2a
# ╠═cb6b271e-3595-4ce4-89c6-1fcc2eaa5136
# ╠═dc3227e9-f40f-4b38-9f01-4e479ce75a48
# ╠═80000554-c2aa-4c96-88d0-c2ac7a452b04
# ╠═008601a2-75b8-4114-ac56-8b52f27735c7
# ╠═08d9207a-7494-4688-90d6-e88548a66da7
# ╠═3fa70e63-ac7b-42e7-87b7-12a0408d0ba6
# ╠═c3b1af35-75ad-40d6-a192-bc03e1f3da92
# ╠═7d640dca-c446-408a-bc2e-422345b17da3
# ╠═39aaebec-177c-4171-9c9e-c0df3517b121
# ╠═79526fb4-3a88-49fc-994a-1bc44c224ccc
# ╠═56ab98d5-fba1-4644-9b9b-2a1197756dbc
# ╠═76a3b35a-8c27-43d3-8148-d8af071bb338
# ╠═7810af6d-d557-42b3-b585-7e80f57747d6
# ╠═5eef1a9e-10fe-4e89-82d5-8df105e7f911
# ╠═95ffc485-1cde-4429-9e4e-9d0062ac57ee
# ╠═fc1777c6-ad4e-4a43-938a-729c5cb84f18
# ╠═b7583245-ed9c-49e4-b8a7-890f814ac3c5
# ╠═27bec401-50f4-4fed-9a5a-b5acd59902ab
# ╠═eff9e5df-ee87-401e-b97a-1a80a3667477
# ╠═b79ed428-994c-4683-ac87-18e453e6a261
# ╠═c17d27c8-dcce-4c11-82d6-3ba3ad5ea66e
# ╠═9fc567d3-5519-4ec1-b749-ec36ceea856b
# ╠═2f790fe7-6b7d-4b7a-8b60-89b416b790e1
# ╠═16154f55-7e5e-4741-9ad8-9ec8d7fc391c
# ╠═11f5be0c-2663-4ca0-adf7-a70203161ed7
# ╠═a5ca7261-198f-4f06-ad1a-fd2870fa0bd4
# ╠═ddd0ff15-c38d-4a56-ad46-99b027b28b10
# ╠═f78b66f8-9601-4ba0-a23a-74fa7e444f43
# ╟─b8715ebf-5290-4307-a554-7fd0ab3086c6
# ╟─715b9bf0-17c5-415c-a1d1-c22d864fd6af
# ╠═15716182-57e9-4f7a-b685-af321fdb8d8a
# ╠═25c18a6a-3e7b-4eab-b38c-b6f6156d65e1
# ╠═56d68aaa-bf87-460d-beb5-420fd7f60fc3
# ╠═ab3577de-6f11-4e9f-b171-f38948b0de09
# ╠═dddde160-19ee-4a91-b22e-e267cbda3ce7
# ╠═eb5f85bd-5324-4e73-91fe-efc42d630176
# ╠═ae35a1df-d594-4de8-946c-de827309cece
# ╠═d811ceb6-b60f-420a-962c-8d6a121588d1
# ╠═444f287e-883f-4f43-a6fd-5c01474d635d
# ╠═209350c1-6cce-452a-87fc-68324672d914
# ╠═cc028ef7-21e5-4666-8ea3-3463c8fb4c4b
# ╠═da17f29a-baf8-4cfb-9aab-f27eac119788
# ╠═a5ce5ea7-8622-4a40-a459-e0728ae162e3
# ╠═d6fa1146-712d-4e2d-a69c-aaadfaf480fa
# ╠═7882d78f-269a-4c44-844c-2dedf18a5199
# ╠═643a399d-b2bd-4bed-afa5-6e6ca4e13dd1
# ╠═04313499-f599-4f73-9c30-a0b214448090
# ╠═930678b5-882d-4042-b087-09bb7561c8f8
# ╠═3cf77b33-2d19-490f-a863-ca376d8c329e
# ╠═bd84e89e-8091-4613-a7d5-f5dd56486819
# ╠═5e59b133-ddae-42b2-86a4-0d880a739b07
# ╠═1f080edd-b6ec-4350-978c-2cd19d5b5860
# ╠═4dc94cfe-f6d1-4310-a444-9f6edfeb0c91
# ╠═55880998-8061-4f3a-b751-116e4901b10b
# ╠═27dc81d6-e986-4a5b-9b56-81e47def9080
# ╠═122a5792-8c9a-4200-b79b-8d32e2f3c3c7
# ╠═36b84ebc-1b5a-496b-8cd2-2e94d7201efd
# ╠═62674e98-3a3a-48b7-860a-012cac18de8a
# ╠═8b71b338-63ad-458b-bc90-6abe3247fddd
# ╠═ce20213f-6a62-494d-82f7-1bfdb83b5191
# ╠═22eb6659-5ea9-4e83-b3fc-6e331e64ddff
# ╠═20411334-d5ae-49f2-abf7-8c03ea4b8d4c
# ╠═e74ea368-ffe4-44cf-92e6-12cd5132986a
# ╠═daf359d5-cd50-48a1-9b79-7e8fe0c3c94d
# ╠═968d82e2-8307-41e9-880e-409c3c405add
# ╠═4d70cb69-5e66-4f72-8fc5-ac8f60abc055
# ╠═e2a1684e-0771-4d17-ba08-c14389f14781
# ╠═d124ac4f-bb0e-4a22-aa15-985d91ebc7c4
# ╠═090bc205-3be5-4c2c-a8a1-fcc7729ef8b6
# ╠═071a38aa-87c8-4ac5-bf42-de61d8de3806
# ╠═5eed1036-5316-4ce6-a65f-000e80e04087
# ╠═df146e27-5d96-4300-bac7-2e496aeea834
# ╠═5e07ffe2-5ce1-4b1e-94bd-624ec13753dd
# ╠═715e8518-bb57-42a7-936d-f4437953c887
# ╠═41e3d9f5-32a4-4942-be5a-ae66c2180612
# ╠═ef43158e-2659-4657-9164-c59ab9735e2e
# ╠═b2907b71-fde3-4e90-8b9e-8c0774db3cb7
# ╠═9264d2c2-ce27-40b9-8c44-5ca25f949641
# ╠═96c5f88c-48a6-4e2e-94f4-ee42d4cd9701
# ╠═64d3324d-c2e8-4255-a158-d315d14664c0
# ╠═1e365da6-9b66-4d5e-9897-e652bf988a76
# ╠═b4e37055-8fe4-4c7d-b790-eff1fcc242a6
# ╠═ad92cff6-b519-44f1-9a8d-699b6ec191b8
# ╠═09b1fdc7-4803-4922-9537-d6ca2a91d630
# ╠═ffbd64f9-1d69-4156-b609-3045db464ecc
# ╠═76ec3dc3-7f7b-4af7-8569-02458f6df245
# ╠═1af67a87-7033-4a83-aa22-860b199bd4f0
# ╠═94b38a9b-2e78-4b56-b55a-15c5424b8267
# ╠═d8c18079-d2d5-4d31-845a-6a176f2084f0
# ╠═f5104a54-fc66-464e-9716-1e266c266133
# ╠═6990831e-caa0-415b-98c6-f031b7653e70
# ╠═35909c90-616d-43f4-bfa1-05a1c841b743
# ╠═3bbed963-a3f2-4a97-8fe0-a5c435986a48
# ╠═be1a1b97-8cbe-4605-993a-cf8d8a9fa759
# ╠═3436a15e-d49e-436c-aeb8-d7937d11ffeb
# ╠═271ed04f-3e14-43ba-969f-bdb63c054c09
# ╠═b1df0dc5-7258-4846-a027-02b215adf976
# ╠═5f262735-7e42-40ef-ae9d-bb7c90f6dcf8
# ╠═6e401f0d-5935-4d6f-bc96-4c40f8b08d9f
# ╠═3d13a62a-51ff-4cf1-bf9e-8228a065088e
# ╠═5c4c39db-8735-40e8-94f9-7731c2baa045
# ╠═4c575172-48e3-4fb2-b373-67c04e5590d8
# ╠═b8fb29c6-27c2-4551-93fd-a65534519b0f
# ╠═6eb38d27-aa50-4309-8f96-7d601cc0277b
# ╠═c058ef32-a609-4ded-85e7-8ada7426c1d3
