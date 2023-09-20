### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 52634c50-c96b-471a-a7f6-4c7dec6ce508
using DrWatson

# ╔═╡ 53afd088-5645-4b1d-9db7-3d8e61a1562d
DrWatson.@quickactivate "Model of DC Differentiation"

# ╔═╡ 7ac5d39e-e6e3-11ea-1250-81b0b1181a87
begin
	using Plots
	using CSV
	using DataFrames
	using Pipe
	using CategoricalArrays
	using CairoMakie
	using AlgebraOfGraphics
end

# ╔═╡ 4935b6f6-8741-11eb-3f64-0bc630593d64
notebook_folder = basename(@__DIR__)

# ╔═╡ 141db2a8-e6e3-11ea-1e2f-4d896be01e9a
md"# $(notebook_folder)"

# ╔═╡ 442de95c-e6e3-11ea-019c-55325a112bed
md"""
Herein, we want to collate and visualise all the data used in this project to give a general idea of what data is available.
"""

# ╔═╡ f9871f5e-8741-11eb-26d4-f7eeaae909f1
md"### Note: Additional datasets"

# ╔═╡ e075029c-8741-11eb-26a6-2312e91367ab
md"Simon and Amit have additional labelling dataset on (pDC), cDC1, and cDC2 in the blood from separated individuals from a previous study on monocyte dynamics."

# ╔═╡ 99aa8ba6-e6e3-11ea-3e42-47d490b85901
md"## Loading experimental data"

# ╔═╡ 4956f8df-b915-4c4b-b48e-852f9a396d02
md"### Labelling data"

# ╔═╡ a458f089-c9f9-4cf5-b632-73c7045a4e01
labelling_data = CSV.read(datadir("exp_raw", "label", "labelling_data.csv"), DataFrame)

# ╔═╡ 231db746-cb28-4a30-9d9a-b4a73856f8a2
labelling_data_d01 = @pipe CSV.read(datadir("exp_raw", "label", "D01_label.csv"), DataFrame) |> transform(_, :enrichment => (x -> (x ./100)), renamecols=false)

# ╔═╡ 19775b22-6635-491f-8337-b75a756fec06
labelling_data_d02 = @pipe CSV.read(datadir("exp_raw", "label", "D02_label.csv"), DataFrame) |> transform(_, :enrichment => (x -> (x ./100)), renamecols=false)

# ╔═╡ 65d8184d-6354-46d3-b0cb-659b4c797659
labelling_data_d04 = @pipe CSV.read(datadir("exp_raw", "label", "D04_label.csv"), DataFrame) |> transform(_, :enrichment => (x -> (x ./100)), renamecols=false)

# ╔═╡ 94e3896b-5eeb-4b69-8584-368b45bce1e4
labelling_data_combined = @pipe vcat(
	labelling_data,
	labelling_data_d01,
	labelling_data_d02,
	labelling_data_d04
) |>
transform(_, [:individual, :population] .=> (x -> string.(x)), renamecols=false) |>
subset(_, :population => (x -> x .!= "Blood"))

# ╔═╡ d9fd135e-e6e8-11ea-0460-bb174c42188c
md"### Glucose enrichment in saliva"

# ╔═╡ 8062c872-e6e8-11ea-0c81-f75f370f4f18
glucose_data_c66 = @pipe CSV.read(datadir("exp_raw","glucose", "C66_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment) |> insertcols!(_, :donor => "C66")

# ╔═╡ a551326e-e6e8-11ea-16c5-e5d4a6b0dab0
glucose_data_c67 = @pipe CSV.read(datadir("exp_raw","glucose", "C67_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment) |> insertcols!(_, :donor => "C67")

# ╔═╡ b79bb84e-e6e8-11ea-3966-0d00108d152e
glucose_data_c68 = @pipe CSV.read(datadir("exp_raw","glucose", "C68_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment) |> insertcols!(_, :donor => "C68")

# ╔═╡ 707e7ae1-90d6-4451-961b-ce72c33b6e23
glucose_data_c52 = @pipe CSV.read(datadir("exp_raw","glucose", "C52_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment)

# ╔═╡ 9ae0078d-d602-4432-af02-aff11227005d
glucose_data_c53 = @pipe CSV.read(datadir("exp_raw","glucose", "C53_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment)

# ╔═╡ e13f7819-ca9d-414f-8bae-836b618f1350
glucose_data_c55 = @pipe CSV.read(datadir("exp_raw","glucose", "C55_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment)

# ╔═╡ e67e1899-1600-4a39-b136-edb1f57bcc18
glucose_data_d01 = @pipe CSV.read(datadir("exp_raw", "glucose", "D01_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment) |> insertcols!(_, :donor => "D01")

# ╔═╡ 4b06410f-3eb1-4a4f-a031-0da585468918
glucose_data_d02 = @pipe CSV.read(datadir("exp_raw", "glucose", "D02_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment) |> insertcols!(_, :donor => "D02")

# ╔═╡ 43fa43ce-2547-4268-853c-e3284ac9cbb5
glucose_data_d04= @pipe CSV.read(datadir("exp_raw", "glucose", "D04_glucose.csv"), DataFrame) |> transform(_, :time => (x -> (x ./ 24.0)) => :time, :enrichment => (x ->(x ./ 100)) => :enrichment) |> insertcols!(_, :donor => "D04")

# ╔═╡ 32997d5a-8742-11eb-1e90-3fefee5473e7
md"#### Combined glucose dataset"

# ╔═╡ f1a63b74-e775-11ea-2ec3-5565299a719c
glucose_data = @pipe vcat(
glucose_data_c66,
glucose_data_c67,
glucose_data_c68,
glucose_data_c52,
glucose_data_c53,
glucose_data_c55,
glucose_data_d01,
glucose_data_d02,
glucose_data_d04) |>
rename(_, :donor => :individual)

# ╔═╡ d6468240-e6e3-11ea-0803-d5cb9a2e6625
md"## Visualise experimental data"

# ╔═╡ f941ddc6-e6e3-11ea-2595-f9d672b2b4a9
md"### Gluscose measurements in saliva"

# ╔═╡ 17b42c87-1598-4596-8f71-e316ae5d761f
@pipe glucose_data |>
groupby(_, :individual) |>
begin
	f_glucose = Figure()
	
	ax_glucose = [Axis(f_glucose[fldmod1(j,3)...], title=first(_[j].individual),xlabel="time (hours)", ylabel="label enrichment",aspect=1) for j in 1:length(_)]
	
	for j in 1:length(_)
		CairoMakie.scatter!(ax_glucose[j], _[j].time, _[j].enrichment) 
	end

	hideydecorations!.(ax_glucose[[(2:3:length(_))..., (3:3:length(_))...]],ticks=false,ticklabels=false)
	
	hidexdecorations!.(ax_glucose[1:6],ticks=false,ticklabels=false)
	
	linkxaxes!(ax_glucose...)
	
	f_glucose
end

# ╔═╡ 0778dca0-e6e4-11ea-2b36-378e3a625756
md"### Cell labelling kinetics in blood"

# ╔═╡ 9a3c029f-9c1e-4503-910b-0ab8013e124c
@pipe labelling_data_combined |>
groupby(_, :individual) |>
begin
	f_labelling = Figure()
	
	ax_labelling = [Axis(f_labelling[fldmod1(j,3)...], title=first(_[j].individual),xlabel="time (days)",ylabel="label enrichment", aspect=1) for j in 1:length(_)]

	celltype_colors = cgrad(:roma, 6, categorical=true)
	celltype_color_dict = Dict("preDC" => celltype_colors[1],"cDC1" => celltype_colors[2], "cDC2" => celltype_colors[3], "pDC" =>celltype_colors[4], "DC2" =>celltype_colors[5], "DC3" =>celltype_colors[6])
	celltype_group_dict = Dict("preDC" => 1,"cDC1" => 2, "cDC2" => 3, "pDC" =>4, "DC2" => 5, "DC3" => 6)
	
	for j in 1:length(_)
		CairoMakie.scatter!(ax_labelling[j],
			_[j].time,
			_[j].enrichment,			
			color=map(x -> celltype_color_dict[x], _[j].population),
			group=map(x -> celltype_group_dict[x], _[j].population)) 
	end
	
	f_labelling[1:2,4] = Legend(f_labelling,[MarkerElement(color = j, marker=:circle) for j in celltype_colors], ["preDC", "cDC1", "cDC2", "pDC", "DC2", "DC3"])

	hideydecorations!.(ax_labelling[[(2:3:length(_))..., (3:3:length(_))...]],ticks=false,ticklabels=false)
	hidexdecorations!.(ax_labelling[1:6],ticks=false,ticklabels=false)

	linkxaxes!(ax_labelling...)
	
	f_labelling
end

# ╔═╡ 30661c72-e6e4-11ea-2608-e9ab333397bc
md"## Save data in concise format to be used later"

# ╔═╡ 8847baba-fc0e-11ea-092b-0907836dda6d
save_data = true ##if true existing data will be overwritten

# ╔═╡ 3fcf90b4-e6e4-11ea-3b2b-51f2749c1f91
begin
	if save_data == true
		save(datadir("exp_pro", "glucose_data.csv"), glucose_data)
		save(datadir("exp_pro", "labelling_data.csv"), labelling_data_combined)
	end
end

# ╔═╡ ba0cb59c-e6e3-11ea-27c0-b73a8909136b
md"#### Package dependecies"

# ╔═╡ 22ee8a6d-1480-43ee-890d-d4bcf4760393
AlgebraOfGraphics.set_aog_theme!()

# ╔═╡ Cell order:
# ╟─141db2a8-e6e3-11ea-1e2f-4d896be01e9a
# ╟─4935b6f6-8741-11eb-3f64-0bc630593d64
# ╟─442de95c-e6e3-11ea-019c-55325a112bed
# ╟─f9871f5e-8741-11eb-26d4-f7eeaae909f1
# ╟─e075029c-8741-11eb-26a6-2312e91367ab
# ╟─99aa8ba6-e6e3-11ea-3e42-47d490b85901
# ╠═4956f8df-b915-4c4b-b48e-852f9a396d02
# ╠═a458f089-c9f9-4cf5-b632-73c7045a4e01
# ╠═231db746-cb28-4a30-9d9a-b4a73856f8a2
# ╠═19775b22-6635-491f-8337-b75a756fec06
# ╠═65d8184d-6354-46d3-b0cb-659b4c797659
# ╠═94e3896b-5eeb-4b69-8584-368b45bce1e4
# ╠═d9fd135e-e6e8-11ea-0460-bb174c42188c
# ╠═8062c872-e6e8-11ea-0c81-f75f370f4f18
# ╠═a551326e-e6e8-11ea-16c5-e5d4a6b0dab0
# ╠═b79bb84e-e6e8-11ea-3966-0d00108d152e
# ╠═707e7ae1-90d6-4451-961b-ce72c33b6e23
# ╠═9ae0078d-d602-4432-af02-aff11227005d
# ╠═e13f7819-ca9d-414f-8bae-836b618f1350
# ╠═e67e1899-1600-4a39-b136-edb1f57bcc18
# ╠═4b06410f-3eb1-4a4f-a031-0da585468918
# ╠═43fa43ce-2547-4268-853c-e3284ac9cbb5
# ╠═32997d5a-8742-11eb-1e90-3fefee5473e7
# ╠═f1a63b74-e775-11ea-2ec3-5565299a719c
# ╠═d6468240-e6e3-11ea-0803-d5cb9a2e6625
# ╠═f941ddc6-e6e3-11ea-2595-f9d672b2b4a9
# ╠═17b42c87-1598-4596-8f71-e316ae5d761f
# ╠═0778dca0-e6e4-11ea-2b36-378e3a625756
# ╠═9a3c029f-9c1e-4503-910b-0ab8013e124c
# ╟─30661c72-e6e4-11ea-2608-e9ab333397bc
# ╠═8847baba-fc0e-11ea-092b-0907836dda6d
# ╠═3fcf90b4-e6e4-11ea-3b2b-51f2749c1f91
# ╟─ba0cb59c-e6e3-11ea-27c0-b73a8909136b
# ╠═52634c50-c96b-471a-a7f6-4c7dec6ce508
# ╠═53afd088-5645-4b1d-9db7-3d8e61a1562d
# ╠═7ac5d39e-e6e3-11ea-1250-81b0b1181a87
# ╠═22ee8a6d-1480-43ee-890d-d4bcf4760393
