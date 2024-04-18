### A Pluto.jl notebook ###
# v0.16.4

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
Herein, we want to collate and visualise all the data used in this project to give a general idea of what data is available. Moreover, we also standardise and harmonise popualtion names, etc. as part of this process.
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
labelling_data_old = @pipe datadir("exp_raw", "label", "labelling_data.csv") |> 
CSV.read(_, DataFrame) |>
transform(_, :population => (x -> replace(x, "preDC" => "ASDC")), renamecols=false)

# ╔═╡ 9678b62d-f2bd-438d-b4b3-8624e40a76fc
labelling_data_new = @pipe datadir("exp_raw", "label") |>
[CSV.read(_*"/"*j*"_label.csv", DataFrame) for j in ["D01", "D02", "D04"]] |>
vcat(_...) |>
subset(_, :population => (x -> x .!= "Blood")) |>
transform(_, :population => (x -> replace(x, "preDC" => "ASDC")), renamecols=false) |>
transform(_, :enrichment => (x -> x ./ 100), renamecols=false)

# ╔═╡ f6a7faa7-8750-424f-ada8-f59ca6554163
labelling_data = vcat(labelling_data_old, labelling_data_new)

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

# ╔═╡ 043c9c36-5636-420d-ae66-78482bbc1e90
glucose_data_d01 = @pipe CSV.read(datadir("exp_raw","glucose", "D01_glucose.csv"), DataFrame) |>
# subset(_, :time => (x -> x .∉ Ref([6.0, 23.0]))) |>
transform(_,
	:time => (x -> (x ./ 24.0)) => :time,
	:enrichment => (x ->(x ./ 100)) => :enrichment) |> 
insertcols!(_, :individual => "D01")

# ╔═╡ 562b739c-49b7-4d2d-857c-82baa48de40b
glucose_data_d01_uncen = @pipe CSV.read(datadir("exp_raw","glucose", "D01_glucose.csv"), DataFrame) |>
transform(_,
	:time => (x -> (x ./ 24.0)) => :time,
	:enrichment => (x ->(x ./ 100)) => :enrichment) |> 
insertcols!(_, :individual => "D01")

# ╔═╡ 4bfa51af-2d97-42af-aace-cb95c881d9a2
glucose_data_d02 = @pipe CSV.read(datadir("exp_raw","glucose", "D02_glucose.csv"), DataFrame) |>
# subset(_, :time => (x -> x .∉ Ref([6.0, 23.0]))) |>
transform(_,
	:time => (x -> (x ./ 24.0)) => :time,
	:enrichment => (x ->(x ./ 100)) => :enrichment) |> 
insertcols!(_, :individual => "D02")

# ╔═╡ c5e2fe6e-4224-4840-9f3e-494ca5a12192
glucose_data_d02_uncen = @pipe CSV.read(datadir("exp_raw","glucose", "D02_glucose.csv"), DataFrame) |>
transform(_,
	:time => (x -> (x ./ 24.0)) => :time,
	:enrichment => (x ->(x ./ 100)) => :enrichment) |> 
insertcols!(_, :individual => "D02")

# ╔═╡ e1106754-21d1-47cd-9271-64dc994a2d08
glucose_data_d04 = @pipe CSV.read(datadir("exp_raw","glucose", "D04_glucose.csv"), DataFrame) |> 
# subset(_, :time => (x -> x .∉ Ref([6.0, 23.0]))) |>
transform(_,
	:time => (x -> (x ./ 24.0)) => :time,
	:enrichment => (x ->(x ./ 100)) => :enrichment) |> 
insertcols!(_, :individual => "D04")

# ╔═╡ 42a5a8b5-de87-4bef-bd00-30f8b6bbb665
glucose_data_d04_uncen = @pipe CSV.read(datadir("exp_raw","glucose", "D04_glucose.csv"), DataFrame) |> 
transform(_,
	:time => (x -> (x ./ 24.0)) => :time,
	:enrichment => (x ->(x ./ 100)) => :enrichment) |> 
insertcols!(_, :individual => "D04")

# ╔═╡ 32997d5a-8742-11eb-1e90-3fefee5473e7
md"#### Combined glucose dataset"

# ╔═╡ f1a63b74-e775-11ea-2ec3-5565299a719c
glucose_data_old = @pipe vcat(
glucose_data_c66,
glucose_data_c67,
glucose_data_c68,
glucose_data_c52,
glucose_data_c53,
glucose_data_c55) |>
rename(_, :donor => :individual)

# ╔═╡ fef75fb4-6d69-4e6d-8c90-34a563af3ae5
glucose_data_new = @pipe vcat(
glucose_data_d01,
glucose_data_d02,
glucose_data_d04)

# ╔═╡ c51ee9d9-0b6a-4168-be29-7e6520b7671f
glucose_data = vcat(glucose_data_old, glucose_data_new)

# ╔═╡ 04adabda-5dee-4710-8bbb-3004e2a6b95b
md"### Cell concentration datasets in the blood and bone marrow"

# ╔═╡ 22851eb3-4aa8-4170-8b36-701d71dff969
md"We load each dataset seperately and calculate absolute cell numbers in blood and bone marrow by multiplying the respective concentrations as measured by flow cytometry with $(blood_vol=5000) and $(bm_vol=1750)."

# ╔═╡ 1647c56f-b774-4160-b25f-987cc539fc61
md"#### New dataset"

# ╔═╡ 7b4b2a9d-6d76-4361-8c4a-3bfd5eff951d
md"##### Blood"

# ╔═╡ 0e3e955c-8a28-4dc1-b167-6452a9d4ac13
df_cell_conc_blood_new = @pipe datadir("exp_raw","cells", "cell_count_blood_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
_ .* blood_vol |>
insertcols!(_,
	:individual => "donor_blood_" .* string.(collect(1:(nrow(_)))),
	:location .=> "blood" 
) |> 
rename(_,
	Symbol("pre-DC") => :ASDC,
	Symbol("CD5+ cDC2") => :DC2,
	Symbol("CD5- DC3") => :DC3
)

# ╔═╡ 31684568-d08a-4c05-8b24-9cd3a01e4d21
md"##### Bone marrow"

# ╔═╡ 88074a62-16fb-4a52-8481-b0dd70aae6e1
df_cell_conc_bm_new = @pipe datadir("exp_raw","cells", "cell_count_bone_marrow_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
_ .* bm_vol |>
insertcols!(_,
	:individual => "donor_bm_" .* string.(collect(1:(nrow(_)))),
	:location .=> "bm" 
)  |> 
rename(_,
	Symbol("pre-DC") => :ASDC,
	Symbol("CD5+ cDC2") => :DC2,
	Symbol("CD5- DC3") => :DC3
)


# ╔═╡ 64c2e8fe-42e6-4b4f-8bbe-7788b52cf99c
md"##### Combined"

# ╔═╡ 7245240c-e760-4f9c-858e-7e9a9d4c8050
df_cell_conc_combined_new  = @pipe vcat(df_cell_conc_blood_new, df_cell_conc_bm_new; cols=:union) |>
insertcols!(_, :dataset .=> "new")

# ╔═╡ fb022ebd-1e9e-4a6b-9f61-3c573507a3c4
md"#### Combined blood and bone marrow dataset"

# ╔═╡ 0c12e7d1-544c-4c74-98cd-c1a883cd88f3
md"Combine all the old and new datasets together:"

# ╔═╡ e5fcca24-ae34-48b3-92bb-c3aa15dcdb5f
df_cell_concentration = df_cell_conc_combined_new

# ╔═╡ 34d155be-363f-4e5f-b677-054326cc3dc2
md"Transform into a long/stacked format:"

# ╔═╡ b8e9b89c-e1a3-4579-af51-04ef654fe085
df_cell_concentration_long = @pipe df_cell_concentration |>
DataFrames.stack(_, variable_name=:population) |>
dropmissing!(_)

# ╔═╡ bcabae34-1b8f-48ea-ad19-62e686c03a46
md"### Cell cycle status datasets in the blood and bone marrow"

# ╔═╡ 3bc01530-07c3-4e43-ac02-313dae3264ce
md"#### Original dataset"

# ╔═╡ e85ffe11-4317-45f8-8dfd-4196e752af9f
md"Next, we transform dataframe into long format, separate the cell state and population information into indvidual columns and translate population names into the updated nomenclature."

# ╔═╡ b52019ff-11f6-4693-a79f-664213a5ed9d
md"#### New dataset"

# ╔═╡ e3c9b3c5-6154-4661-9c6d-f49c84615453
md"The new dataset contains the data for ASDC and cDC1 from the dataset part of original submission. Thus, we will just use this new dataset for the analysis of the revised manuscript submission."

# ╔═╡ 50eeb30e-8d6e-4375-b991-1112c66f9668
md"##### Blood"

# ╔═╡ 6c6fde96-c4cc-477b-aa4a-741e7985a8a1
df_cycle_blood_new = @pipe datadir("exp_raw","cycle", "cell_cycle_blood_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
insertcols!(_,
	:individual => "donor_blood_" .* string.(collect(1:nrow(_))),
	:location .=> "blood" 
)

# ╔═╡ a022b119-d994-4b79-bbb8-a5378518d59c
md"##### Bone marrow"

# ╔═╡ 3b5e3aef-e1c5-4d10-a3ab-0111ecc6c323
df_cycle_bm_new = @pipe datadir("exp_raw","cycle", "cell_cycle_BM_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
insertcols!(_,
	:individual => "donor_bm_" .* string.(collect(1:nrow(_))),
	:location .=> "bm" 
)

# ╔═╡ d518a934-5ff1-4e95-ab40-db47bfb94070
md"Next, we transform the new dataframe into long format, separate the cell state and population information into indvidual columns and make sure that all population names conform with the updated nomenclature."

# ╔═╡ 484eac7f-edf8-4443-8f82-1ff4eede2682
df_cycle_long_new = @pipe vcat(df_cycle_blood_new, df_cycle_bm_new) |>
stack(_, variable_name=:measurement) |>
dropmissing!(_) |>
transform(_,
	:measurement => ByRow((x) -> match(r"(ASDC|cDC1|DC1|DC23|DC2|DC3|dc3|pDC|DC)(.*)", x).captures[1]) => :population,
	:measurement => ByRow((x) -> match(r"(ASDC|cDC1|DC1|DC23|DC2|DC3|dc3|pDC|DC)\s?(.*)", x).captures[2]) => :state) |> 
transform(_, :population => (x -> replace(x, "DC1" => "cDC1", "DC" => "cDC1", "DC23" => "DC3", "dc3" => "DC3")) => :population,
	:state => (x -> replace(x, "Go" => "G0", "G2SM" => "G2", "g2sm" => "G2")) => :state) |>
select(_, Not(:measurement)) |>
insertcols!(_, :dataset .=> "combined")

# ╔═╡ ef7a2d01-b624-4d6e-9025-c64efbd0db75
md"#### Combine the original and new cell cycle datasets"

# ╔═╡ b2a613fe-fb87-4aa8-9ed2-42f43452fcdc
df_cycle_long = df_cycle_long_new

# ╔═╡ d6468240-e6e3-11ea-0803-d5cb9a2e6625
md"## Visualise experimental data"

# ╔═╡ f941ddc6-e6e3-11ea-2595-f9d672b2b4a9
md"### Gluscose measurements in blood"

# ╔═╡ 17b42c87-1598-4596-8f71-e316ae5d761f
@pipe glucose_data |>
groupby(_, :individual) |>
begin
	f_glucose = Figure()
	
	ax_glucose = [Axis(f_glucose[fldmod1(j,3)...], title=first(_[j].individual),xlabel="time (hours)", ylabel="labelled glucose enrichment (in blood)",aspect=1) for j in 1:length(_)]
	
	for j in 1:length(_)
		CairoMakie.lines!(ax_glucose[j], _[j].time, _[j].enrichment)
		CairoMakie.scatter!(ax_glucose[j], _[j].time, _[j].enrichment)
	end
	[hidexdecorations!(j; ticklabels=false, ticks=false) for j in  ax_glucose[1:6]]
	[hideydecorations!(j; ticklabels=false, ticks=false) for j in  ax_glucose[[1,2,3,5,6,7,8,9]]]
	linkxaxes!(ax_glucose...)
	
	f_glucose
end

# ╔═╡ 0778dca0-e6e4-11ea-2b36-378e3a625756
md"### Cell labelling kinetics in blood"

# ╔═╡ 9a3c029f-9c1e-4503-910b-0ab8013e124c
@pipe labelling_data |>
groupby(_, :individual) |>
begin
	f_labelling = Figure()
	
	ax_labelling = [Axis(f_labelling[fldmod1(j,3)...], title=first(_[j].individual),xlabel="time (days)",ylabel="label enrichment", aspect=1) for j in 1:length(_)]
	

	pop_cgrad = cgrad(:roma, 6, categorical=true)
	pop_group_dict = Dict("ASDC" => 1,"cDC1" => 2, "cDC2" => 3, "pDC" =>4, "DC2" =>5, "DC3" =>6)
	pop_cgrad_dict = Dict([k => pop_cgrad[v] for (k, v) in pop_group_dict])
	
	for j in 1:length(_)
		CairoMakie.scatter!(ax_labelling[j],
			_[j].time,
			_[j].enrichment,			
			color=map(x -> pop_cgrad_dict[x], _[j].population),
			group=map(x -> pop_group_dict[x], _[j].population)) 
	end
	
	f_labelling[1:2,4] = Legend(f_labelling,[MarkerElement(color = j, marker=:circle) for j in pop_cgrad[[values(pop_group_dict)...]]], [keys(pop_group_dict)...] )
	
	
	f_labelling
end

# ╔═╡ 4dc25ab4-2d41-44b5-bf87-314e886041d7
md"### Cell popualtion concentrations across blood and bone marrow"

# ╔═╡ 3e8a8a7d-e593-4a8b-9371-9f7ee5f855bd
@pipe df_cell_concentration_long |>
data(_) * 
mapping(:dataset,:value => "# cells", color = :dataset, row=:population, col=:location) * 
visual(BoxPlot) |>
draw(_; facet =(; linkyaxes=:none))

# ╔═╡ 06ce9870-1bdc-4264-b1b0-b644ef83b764
md"It appears that the new dataset has consistently lower DC cell concentrations compared to the original dataset. After consultating with Simon, he confirmed that the original data was measured using counting beads while the new dataset used the inherent capability of the flow cytometer to count cells per defined volume. The latter approach should yield more accurate numbers and thus will be used in the subsequent step to determine the cell compartment sizes and ratio."

# ╔═╡ 77ee7ac7-9d16-4a52-9aec-aa1c681eb3ff
md"### Cell population cell cycle status across blood and bone marrow"

# ╔═╡ d767393d-8ee8-464f-b36a-4a76232ab748
@pipe df_cycle_long |>
subset(_, :state => (x -> x .== "G2")) |>
data(_) * 
mapping(:population,:value => "% in SG2M phase", color = :dataset, dodge=:dataset, col=:location) * 
visual(BoxPlot) |>
draw(_; facet =(; linkyaxes=:minimal))

# ╔═╡ 30661c72-e6e4-11ea-2608-e9ab333397bc
md"## Save data in concise format to be used later"

# ╔═╡ 8847baba-fc0e-11ea-092b-0907836dda6d
save_data = true ##if true existing data will be overwritten

# ╔═╡ 3fcf90b4-e6e4-11ea-3b2b-51f2749c1f91
begin
	if save_data == true
		save(datadir("exp_pro", "glucose_data_revision.csv"), glucose_data)
		save(datadir("exp_pro", "labelling_data_revision.csv"), labelling_data)
		save(datadir("exp_pro", "cell_concentration_data_revision.csv"), df_cell_concentration_long)
		save(datadir("exp_pro", "cell_cycle_data_revision.csv"), df_cycle_long)
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
# ╟─4956f8df-b915-4c4b-b48e-852f9a396d02
# ╠═a458f089-c9f9-4cf5-b632-73c7045a4e01
# ╠═9678b62d-f2bd-438d-b4b3-8624e40a76fc
# ╠═f6a7faa7-8750-424f-ada8-f59ca6554163
# ╠═d9fd135e-e6e8-11ea-0460-bb174c42188c
# ╠═8062c872-e6e8-11ea-0c81-f75f370f4f18
# ╠═a551326e-e6e8-11ea-16c5-e5d4a6b0dab0
# ╠═b79bb84e-e6e8-11ea-3966-0d00108d152e
# ╠═707e7ae1-90d6-4451-961b-ce72c33b6e23
# ╠═9ae0078d-d602-4432-af02-aff11227005d
# ╠═e13f7819-ca9d-414f-8bae-836b618f1350
# ╠═043c9c36-5636-420d-ae66-78482bbc1e90
# ╠═562b739c-49b7-4d2d-857c-82baa48de40b
# ╠═4bfa51af-2d97-42af-aace-cb95c881d9a2
# ╠═c5e2fe6e-4224-4840-9f3e-494ca5a12192
# ╠═e1106754-21d1-47cd-9271-64dc994a2d08
# ╠═42a5a8b5-de87-4bef-bd00-30f8b6bbb665
# ╠═32997d5a-8742-11eb-1e90-3fefee5473e7
# ╠═f1a63b74-e775-11ea-2ec3-5565299a719c
# ╠═fef75fb4-6d69-4e6d-8c90-34a563af3ae5
# ╠═c51ee9d9-0b6a-4168-be29-7e6520b7671f
# ╟─04adabda-5dee-4710-8bbb-3004e2a6b95b
# ╟─22851eb3-4aa8-4170-8b36-701d71dff969
# ╟─1647c56f-b774-4160-b25f-987cc539fc61
# ╟─7b4b2a9d-6d76-4361-8c4a-3bfd5eff951d
# ╠═0e3e955c-8a28-4dc1-b167-6452a9d4ac13
# ╟─31684568-d08a-4c05-8b24-9cd3a01e4d21
# ╠═88074a62-16fb-4a52-8481-b0dd70aae6e1
# ╟─64c2e8fe-42e6-4b4f-8bbe-7788b52cf99c
# ╠═7245240c-e760-4f9c-858e-7e9a9d4c8050
# ╟─fb022ebd-1e9e-4a6b-9f61-3c573507a3c4
# ╟─0c12e7d1-544c-4c74-98cd-c1a883cd88f3
# ╠═e5fcca24-ae34-48b3-92bb-c3aa15dcdb5f
# ╟─34d155be-363f-4e5f-b677-054326cc3dc2
# ╠═b8e9b89c-e1a3-4579-af51-04ef654fe085
# ╟─bcabae34-1b8f-48ea-ad19-62e686c03a46
# ╟─3bc01530-07c3-4e43-ac02-313dae3264ce
# ╟─e85ffe11-4317-45f8-8dfd-4196e752af9f
# ╟─b52019ff-11f6-4693-a79f-664213a5ed9d
# ╟─e3c9b3c5-6154-4661-9c6d-f49c84615453
# ╟─50eeb30e-8d6e-4375-b991-1112c66f9668
# ╠═6c6fde96-c4cc-477b-aa4a-741e7985a8a1
# ╟─a022b119-d994-4b79-bbb8-a5378518d59c
# ╠═3b5e3aef-e1c5-4d10-a3ab-0111ecc6c323
# ╟─d518a934-5ff1-4e95-ab40-db47bfb94070
# ╠═484eac7f-edf8-4443-8f82-1ff4eede2682
# ╟─ef7a2d01-b624-4d6e-9025-c64efbd0db75
# ╠═b2a613fe-fb87-4aa8-9ed2-42f43452fcdc
# ╟─d6468240-e6e3-11ea-0803-d5cb9a2e6625
# ╟─f941ddc6-e6e3-11ea-2595-f9d672b2b4a9
# ╠═17b42c87-1598-4596-8f71-e316ae5d761f
# ╟─0778dca0-e6e4-11ea-2b36-378e3a625756
# ╟─9a3c029f-9c1e-4503-910b-0ab8013e124c
# ╟─4dc25ab4-2d41-44b5-bf87-314e886041d7
# ╠═3e8a8a7d-e593-4a8b-9371-9f7ee5f855bd
# ╟─06ce9870-1bdc-4264-b1b0-b644ef83b764
# ╟─77ee7ac7-9d16-4a52-9aec-aa1c681eb3ff
# ╠═d767393d-8ee8-464f-b36a-4a76232ab748
# ╟─30661c72-e6e4-11ea-2608-e9ab333397bc
# ╠═8847baba-fc0e-11ea-092b-0907836dda6d
# ╠═3fcf90b4-e6e4-11ea-3b2b-51f2749c1f91
# ╟─ba0cb59c-e6e3-11ea-27c0-b73a8909136b
# ╠═52634c50-c96b-471a-a7f6-4c7dec6ce508
# ╠═53afd088-5645-4b1d-9db7-3d8e61a1562d
# ╠═7ac5d39e-e6e3-11ea-1250-81b0b1181a87
# ╠═22ee8a6d-1480-43ee-890d-d4bcf4760393
