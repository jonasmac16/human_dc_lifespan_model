### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 8ad38cac-1a4e-459e-8712-8c3a58877ff1
using DrWatson

# ╔═╡ e10fa7f7-0bd8-4c7a-a608-ad9092caae23
DrWatson.@quickactivate "Model of DC Differentiation"

# ╔═╡ d22ae2f4-2b1d-11eb-1a18-ff8b94d1d0c1
begin
	using ExcelFiles
	using DataFrames
	using DataFramesMeta
	using Plots
	using StatsPlots
	using Statistics
	using StatsBase
	using Turing
	using Distributions
	using GalacticOptim
	using Optim
	using BlackBoxOptim
	using Pipe
	using CategoricalArrays
	using AlgebraOfGraphics
	using CairoMakie
	using Pipe
	using CSV
	using Random
end

# ╔═╡ 11944318-873a-11eb-1661-f56a06d1cb4c
begin
	notebook_dir = @__DIR__
	notebook_dir_name = basename(notebook_dir)
end

# ╔═╡ b54b0f2e-2b1d-11eb-0334-f32dab087681
md"## $(notebook_dir_name)"

# ╔═╡ e007a4a2-2b1d-11eb-3649-91130111debc
md"## Purpose"

# ╔═╡ dd13ffa2-2b1d-11eb-00db-47d4a6ad1305
md"""
Estimating proliferation rate priors and intra- and inter-compartment population size ratios.
"""

# ╔═╡ dab7bec4-2b1d-11eb-2ff4-034c27c6c5f1
md"## Data"

# ╔═╡ d76acba8-7690-11eb-37b5-f903fb128ee0
processed_data_folder = datadir("exp_pro");

# ╔═╡ 3a59091e-2b21-11eb-13e1-9707f83f67de
md"The `processed` data is located here $(processed_data_folder)"

# ╔═╡ 73e87950-2b21-11eb-22b0-f70d8354e6c0
md"## Data Input"

# ╔═╡ e6a0a11c-9eb7-4fce-bcad-14fccfc33421
md"We load the combined data of the original manuscript and the newly acquired dataset for the paper revision below:"

# ╔═╡ a7177ccc-2b21-11eb-26e8-dd349d40f1f0
md"### Cell number concentration"

# ╔═╡ 7ce6e982-d06f-4ac2-a8a8-3bb979e695e0
df_cell_concentration = @pipe datadir("exp_pro", "cell_concentration_data_revision.csv") |>
CSV.read(_, DataFrame) |>
subset(_, :dataset => (x -> x .== "new"))

# ╔═╡ ba837004-2b21-11eb-33f9-efd996b7f6e8
md"### Cell cycle status"

# ╔═╡ 57200dd2-09b4-4bac-b2ce-f8ccb5b0b2d3
df_cell_cycle = @pipe datadir("exp_pro", "cell_cycle_data_revision.csv") |>
CSV.read(_, DataFrame)

# ╔═╡ 39e8839c-7549-11eb-0e7e-efd773f2441a
md"## Analyse and summarise data"

# ╔═╡ 39c9d2ba-7549-11eb-2722-c3d6c5a4a836
md"First, we visualise the cell cycle and cell number measurements in both compartments:"

# ╔═╡ 7d4edb59-a104-4b66-906d-e77b4aced31b
begin
	renamer_location = renamer("blood" => "blood", "bm" => "bone marrow")
	fig_sg2m = Figure(resolution=(700,400))
	subfig = fig_sg2m[1,1]
	
	
	ax_sg2m = @pipe df_cell_cycle |>
		subset(_, :state => (x -> x .== "G2"))  |> transform(_, :population => (x -> categorical(x, levels=["ASDC", "cDC1", "cDC2","DC2", "DC3", "pDC"])), renamecols=false) |>
		transform(_, :location => (x -> replace(x, "blood" => "blood", "bm" => "bone marrow")), renamecols=false) |>
		transform(_, :location => (x -> categorical(x, levels=["bone marrow", "blood"])), renamecols=false) |>
		data(_) * mapping(:population, :value, layout=:location) *(visual(BoxPlot, outliers=false)*mapping(color=:population) + visual(Scatter, color=:black)) |>
		draw!(subfig, _; axis=(aspect=1,),  palettes = (color = [colorant"#755494",colorant"#de3458" ,colorant"#4e65a3", colorant"#c8ab37ff"],))
		ax_sg2m[1].axis.ylabel = "% of subset in SG2M phase"
		fig_sg2m

end

# ╔═╡ 6bc30ed8-32fe-4a4f-a522-113dba6d5837
begin
	renamer_location_pop = renamer("blood" => "blood", "bm" => "bone marrow")
	fig_cell_number = Figure(resolution=(700,400))
	# ax = Axis(fig_sg2m[1, 1], title="Some plot")
	subfig_cell_number = fig_cell_number[1,1] #[Axis(fig_sg2m[1,j]) for j in 1:3]
	
@pipe df_cell_concentration |>
	transform(_, :population => (x -> categorical(x, levels=["ASDC", "cDC1", "cDC2", "DC2", "DC3", "pDC"])), renamecols=false) |>
	transform(_, :location => (x -> replace(x, "blood" => "blood", "bm" => "bone marrow")), renamecols=false) |>
	transform(_, :location => (x -> categorical(x, levels=["bone marrow", "blood"])), renamecols=false) |>
	data(_) * mapping(:population, :value, layout=:location) *(visual(BoxPlot, outliers=false)*mapping(color=:population) + visual(Scatter, color=:black)) |> 
	draw!(subfig_cell_number,_; axis=(ylabel="# cells in compartment",aspect=1,), palettes = (color = [colorant"#755494",colorant"#de3458" ,colorant"#4e65a3", colorant"#c8ab37ff"],))
	fig_cell_number
end

# ╔═╡ a2e62d9a-7691-11eb-2a67-75febee39da9
md"## Analyse cell number data and calculate cell ratios"

# ╔═╡ b859d2da-7691-11eb-3c4b-23aaf806d05c
md"First, we calculate the intra-compartment ratios for each donor individually and then summarise the individual ratios."

# ╔═╡ ec6f9ad2-7691-11eb-1f9a-1f4ea84d92d7
df_ratios = @linq df_cell_concentration |> where(:population .!= "pDC") |> groupby(:individual) |> transform(ratio = first(:value)./:value)

# ╔═╡ 0a981ce6-7697-11eb-30d4-318124f079a5
begin
	df_ratios_intra = @linq df_ratios |> 
	DataFrames.subset(:population => (x -> x .∈ Ref(["ASDC", "cDC1", "DC2", "DC3"]))) |> 
	DataFrames.select(Not(:value)) |> 
	groupby([:location, :population]) |> 
	DataFrames.combine(:ratio => (x -> [mean(x) median(x) std(x) minimum(x) maximum(x)] )=> [:mean, :median, :sd, :min, :max])
end

# ╔═╡ 708537f8-76b7-11eb-2cb2-d19676df1dfd
# begin
# 	RpreDCcDC1b_mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array)[1]
# 	RpreDCcDC1bm_mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array)[1]
# 	RpreDCcDC2b_mean = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:mean) |> Array)[1]
# 	RpreDCcDC2bm_mean = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:mean) |> Array)[1]
	
# 	RpreDCcDC1b_median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array)[1]
# 	RpreDCcDC1bm_median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array)[1]
# 	RpreDCcDC2b_median = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:median) |> Array)[1]
# 	RpreDCcDC2bm_median = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:median) |> Array)[1]
	
# 	RpreDCcDC1b_min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array)[1]
# 	RpreDCcDC1bm_min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array)[1]
# 	RpreDCcDC2b_min = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:min) |> Array)[1]
# 	RpreDCcDC2bm_min = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:min) |> Array)[1]
	
# 	RpreDCcDC1b_max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array)[1]
# 	RpreDCcDC1bm_max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array)[1]
# 	RpreDCcDC2b_max = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:max) |> Array)[1]
# 	RpreDCcDC2bm_max = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:max) |> Array)[1];
# end

# ╔═╡ 520dbc20-fdd8-451a-80fb-8e2d3232466f
begin
	RASDCcDC1b = (;
		mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array)[1],
		median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array)[1], 
		min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array)[1], 
		max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array)[1])
	RASDCcDC1bm = (;
		mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array)[1],
		median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array)[1],
		min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array)[1],
		max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array)[1])
	RASDCDC2b = (;
		mean = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:mean) |> Array)[1],
		median = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:median) |> Array)[1],
		min = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:min) |> Array)[1],
		max = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:max) |> Array)[1])
	RASDCDC2bm = (;
		mean = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:mean) |> Array)[1],
		median = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:median) |> Array)[1],
		min = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:min) |> Array)[1],
		max = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:max) |> Array)[1])
end

# ╔═╡ 27af2e26-7b7b-11eb-1eaa-f5cabf39942a
md"In order to identify the most reasonable population to base our cross-compartment calculation on (following section), we also determine the variability of each population in the both compartments:"

# ╔═╡ 80290752-7b7b-11eb-2a1d-5707c650a0b0
df_cell_vari = @linq df_cell_concentration |>
where(:population .∈ Ref(["preDC", "cDC1", "DC2", "DC3"])) |>
groupby([:location, :population]) |>
DataFrames.combine(:value =>(x -> [mean(x) median(x) std(x) minimum(x) maximum(x)] )=> [:mean, :median, :sd, :min, :max])

# ╔═╡ 8007f76e-7753-11eb-2d1e-49006c5fa6f2
md"### Calculating intercompartment ratios"

# ╔═╡ 310fcb44-769c-11eb-27f9-139e551fa012
md"""
The following ratios can be directly calculated per individual and thus we can assess and incorporate the inter-individual variability straightforwardly:

**RASDCcDC1bm** = ASDCbm / cDC1bm

**RASDCDC2bm** = ASDCbm / DC2bm 

**RASDCcDC1b** = ASDCb / cDC1b

**RASDCDC2b** = ASDCb / DC2b
"""



# ╔═╡ 5db2582e-774b-11eb-076b-81fab3b405fd
md"""
Due to the lack of data from both bone marrow and blood from the same individual we use population sizes in the two compartments assessed in independent donors. This can be achieved via multiple routes. One can calculate a single accross compartment ratio and derive the remaining number (Approach 1) or one can calculate all ratios directly from the data (Approach 2). The first approach can be achieved via 3 calculation depending which accross compartment ratio one calculates first:

**Approach 1a:**

**RASDC** = ASDCbm/ASDCb

**RcDC1** = RASDC * (RASDCcDC1b/RASDCcDC1bm)

**RDC2** = RASDC * (RASDCDC2b/RASDCDC2bm)


**Approach 1b:**

**RcDC1** = cDC1bm/cDC1b

**RASDC** = RcDC1 * RASDCcDC1bm/RASDCcDC1b

**RDC2** = RASDC * (RASDCDC2b/RASDCDC2bm)



**Approach 1c:**

**RDC2** = DC2Cbm/DC2b

**RASDC** = RDC2 * RASDCDC2bm/RASDCDC2b

**RcDC1** = RASDC * (RASDCcDC1b/RASDCcDC1bm)


**Approach 2:**

**RASDC** = ASDCbm/ASDCb

**RcDC1** = cDC1bm/cDC1b

**RDC2** = DC2Cbm/DC2b


"""

# ╔═╡ 6f296b4c-774b-11eb-09c1-55b81908299a
md"""
The above equations are derived as follows:

**RpreDC = ASDCbm/ASDCb**

RASDC = (RASDCcDC1bm *cDC1bm) / (RASDCcDC1b*cDC1b)

RASDC = cDC1bm/cDC1b * RASDCcDC1bm/RASDCcDC1b

**RASDC = RcDC1 * RASDCcDC1bm/RASDCcDC1b**

RASDC = (RASDCDC2bm *DC2bm) / (RASDCDC2b*DC2b)

RASDC = DC2bm/DC2b * RASDCDC2bm/RASDCDC2b

**RASDC = RDC2 * RASDCDC2bm/RASDCDC2b**


-----------------------

**RcDC1 = cDC1bm/cDC1b**

RcDC1 = (ASDCbm/RASDCcDC1bm) / (ASDCb/RASDCcDC1b)

RcDC1 = (ASDCbm/RASDCcDC1bm) * (RASDCcDC1b/ASDCb)

RcDC1 = (ASDCbm/ASDCb) * (RASDCcDC1b/RASDCcDC1bm)

**RcDC1 = RASDC * (RASDCcDC1b/RASDCcDC1bm)**


------------------------

**RDC2 = DC2Cbm/DC2b**

**RDC2 = RASDC * (RASDCDC2b/RASDCDC2bm)**

"""

# ╔═╡ 309dd70c-7753-11eb-1777-47b2a12efaf1
md"Following the outlined equations above we are calculating the intercompartment ratios for all 4 approach and contrast and compare the results below:"

# ╔═╡ 9967f735-b5e2-459a-b75c-43468af2baf1
df_tmp = @linq df_cell_concentration |>
where(:population .∈ Ref(["ASDC", "cDC1", "DC2", "DC3"])) |>
groupby([:population, :location]) |>
DataFrames.combine(:value => (x -> [mean(x) median(x) minimum(x) maximum(x)] )=> [:mean, :median, :min, :max])

# ╔═╡ 0436f3ba-769c-11eb-1c45-fd1dc9a8d75d
begin
	## Approach 1a
	# R_preDC_mean = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	# R_preDC_median = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	# R_preDC_min = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	# R_preDC_max = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	R_ASDC = (;
		mean = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1],
		median = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1], 
		min = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1], 
		max = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1] 
	)
	
	RcDC1 = (;
	mean = R_ASDC.mean * (RASDCcDC1b.mean/RASDCcDC1bm.mean),
	median = R_ASDC.median * (RASDCcDC1b.median/RASDCcDC1bm.median),
	min = R_ASDC.min * (RASDCcDC1b.min/RASDCcDC1bm.max),
	max = R_ASDC.max * (RASDCcDC1b.max/RASDCcDC1bm.min)
	)
	
	RDC2 = (;
	mean = R_ASDC.mean * (RASDCDC2b.mean/RASDCDC2bm.mean),
	median = R_ASDC.median * (RASDCDC2b.mean/RASDCDC2bm.median),
	min = R_ASDC.min * (RASDCDC2b.min/RASDCDC2bm.max),
	max = R_ASDC.max * (RASDCDC2b.max/RASDCDC2bm.min)
	)
	
	df_base = DataFrame(
		RASDC_cDC1_blood_mean = RASDCcDC1b.mean,
		RASDC_cDC1_bm_mean = RASDCcDC1bm.mean,
		RASDC_DC2_blood_mean = RASDCDC2b.mean,
		RASDC_DC2_bm_mean = RASDCDC2bm.mean,
		RASDC_cDC1_blood_median = RASDCcDC1b.median,
		RASDC_cDC1_bm_median = RASDCcDC1bm.median,
		RASDC_DC2_blood_median = RASDCDC2b.median,
		RASDC_DC2_bm_median = RASDCDC2bm.median,
		RASDC_cDC1_blood_min = RASDCcDC1b.min,
		RASDC_cDC1_bm_min = RASDCcDC1bm.min,
		RASDC_DC2_blood_min = RASDCDC2b.min,
		RASDC_DC2_bm_min = RASDCDC2bm.min,
		RASDC_cDC1_blood_max = RASDCcDC1b.max,
		RASDC_cDC1_bm_max = RASDCcDC1bm.max,
		RASDC_DC2_blood_max = RASDCDC2b.max,
		RASDC_DC2_bm_max = RASDCDC2bm.max)
	
	df_0 = hcat(
		df_base,
		DataFrame(
			RASDC_mean = R_ASDC.mean,
			RASDC_median = R_ASDC.median,
			RASDC_min = R_ASDC.min,
			RASDC_max = R_ASDC.max,
			RcDC1_mean = RcDC1.mean,
			RcDC1_median = RcDC1.median,
			RcDC1_min = RcDC1.min,
			RcDC1_max = RcDC1.max,
			RDC2_mean = RDC2.mean,
			RDC2_median = RDC2.median,
			RDC2_min = RDC2.min,
			RDC2_max = RDC2.max
		)
	)

df_all_ratios = @linq DataFrames.stack(df_0) |>
	DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> 
	DataFrames.select(Not(:variable))
end

# ╔═╡ 5749840a-774c-11eb-2fb4-afef062e407f
begin
	## Approach 1b
	RcDC1_1 = (;
		mean = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1],
		median = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1],
		min = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1],
		max = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	)
	
	R_ASDC_1 = (;
		mean = RcDC1_1.mean * (RASDCcDC1bm.mean/RASDCcDC1b.mean),
		median = RcDC1_1.median * (RASDCcDC1bm.median/RASDCcDC1b.median),
		min = RcDC1_1.min * (RASDCcDC1bm.min/RASDCcDC1b.max),
		max = RcDC1_1.max * (RASDCcDC1bm.max/RASDCcDC1b.min)
	)
	
	RDC2_1 = (;
		mean = R_ASDC_1.mean * (RASDCDC2b.mean/RASDCDC2bm.mean),
		median = R_ASDC_1.median * (RASDCDC2b.median/RASDCDC2bm.median),
		min = R_ASDC_1.min * (RASDCDC2b.min/RASDCDC2bm.max),
		max = R_ASDC_1.max * (RASDCDC2b.max/RASDCDC2bm.min)
	)
	
	df_1 = hcat(
		df_base,
		DataFrame(
			RASDC_mean = R_ASDC_1.mean,
			RASDC_median = R_ASDC_1.median,
			RASDC_min = R_ASDC_1.min,
			RASDC_max = R_ASDC_1.max,
			RcDC1_mean = RcDC1_1.mean,
			RcDC1_median = RcDC1_1.median,
			RcDC1_min = RcDC1_1.min,
			RcDC1_max = RcDC1_1.max,
			RDC2_mean = RDC2_1.mean,
			RDC2_median = RDC2_1.median,
			RDC2_min = RDC2_1.min,
			RDC2_max = RDC2_1.max
		)
	)

	df_all_ratios_1 = @linq DataFrames.stack(df_1) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 5edce4f2-774c-11eb-16f7-ffcfe2e6ff34
begin
	## Aproach 1c
	RDC2_2 = (;
		mean = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1],
		median = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1],
		min = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1],
		max = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	)
	
	R_ASDC_2 = (;
		mean = RDC2_2.mean * (RASDCDC2bm.mean/RASDCDC2b.mean),
		median = RDC2_2.median * (RASDCDC2bm.median/RASDCDC2b.median),
		min = RDC2_2.min * (RASDCDC2bm.min/RASDCDC2b.max),
		max = RDC2_2.max * (RASDCDC2bm.max/RASDCDC2b.min)
	)
	
	RcDC1_2 = (;		
		mean = R_ASDC_2.mean * (RASDCcDC1b.mean/RASDCcDC1bm.mean),
		median = R_ASDC_2.median * (RASDCcDC1b.median/RASDCcDC1bm.median),
		min = R_ASDC_2.min * (RASDCcDC1b.min/RASDCcDC1bm.max),
		max = R_ASDC_2.max * (RASDCcDC1b.max/RASDCcDC1bm.min)
	)
		
	df_2 = hcat(
		df_base,
		DataFrame(
			RASDC_mean = R_ASDC_2.mean,
			RASDC_median = R_ASDC_2.median,
			RASDC_min = R_ASDC_2.min,
			RASDC_max = R_ASDC_2.max,
			RcDC1_mean = RcDC1_2.mean,
			RcDC1_median = RcDC1_2.median,
			RcDC1_min = RcDC1_2.min,
			RcDC1_max = RcDC1_2.max,
			RDC2_mean = RDC2_2.mean,
			RDC2_median = RDC2_2.median,
			RDC2_min = RDC2_2.min,
			RDC2_max = RDC2_2.max
		)
	)
	df_all_ratios_2 = @linq DataFrames.stack(df_2) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 2c7b6d3a-774c-11eb-354d-85b49dd83eeb
begin
	## Approach 2
	R_ASDC_3 = (;
		mean = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1],
		median = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1],
		min = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1],
		max = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	)
	
	RcDC1_3 = (;
		mean = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1],
		median = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1],
		min = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1],
		max = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	)
	
	RDC2_3 = (;
		mean = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1],
		median = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1],
		min = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1],
	 	max = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	)
		
	df_3 = hcat(
		df_base, 
		DataFrame(
			RASDC_mean = R_ASDC_3.mean,
			RASDC_median = R_ASDC_3.median,
			RASDC_min = R_ASDC_3.min,
			RASDC_max = R_ASDC_3.max,
			RcDC1_mean = RcDC1_3.mean,
			RcDC1_median = RcDC1_3.median,
			RcDC1_min = RcDC1_3.min,
			RcDC1_max = RcDC1_3.max,
			RDC2_mean = RDC2_3.mean,
			RDC2_median = RDC2_3.median,
			RDC2_min = RDC2_3.min,
			RDC2_max = RDC2_3.max
		)
	)

	df_all_ratios_3 = @linq DataFrames.stack(df_3) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 24c1813f-bad7-47ba-a581-32c13ce4a451
md"### Combine all results"

# ╔═╡ 88ce1361-0a57-4180-8b2c-9125801c51b1
begin
	function rename_ratios(rnames)
		mapping = ["RASDC" => "R_ASDC",
				"RcDC1" => "R_cDC1",
				"RDC2" => "R_DC2",
				"RASDC_cDC1_bm" => "R_ASDCcDC1bm",
				"RASDC_DC2_bm" => "R_ASDCDC2bm",
				"RASDC_cDC1_blood" => "R_ASDCcDC1b",
				"RASDC_DC2_blood" => "R_ASDCDC2b",
				"RDC3" => "R_DC3"]
		
		return replace(rnames, mapping...)
	end


	df_ratio_approaches_combined = @pipe vcat(transform(df_all_ratios, :value => (x -> [1 for j in x]) => :method),
transform(df_all_ratios_1, :value => (x -> [2 for j in x]) => :method),
transform(df_all_ratios_2, :value => (x -> [3 for j in x]) => :method),
transform(df_all_ratios_3, :value => (x -> [4 for j in x]) => :method)) |>
transform(_, :method => (x -> string.(x)), renamecols=false) |>
transform(_, :method => (x -> replace(x, "1"=>"1a", "2"=>"1b", "3"=>"1c", "4"=>"2")), renamecols=false) |>
transform(_, :method => (x -> categorical(x, levels=["1a", "1b", "1c","2"])), renamecols=false)

df_ratio_approaches_combined_wdc3 = @pipe df_cell_concentration |>
subset(_, :population => x -> x .== "DC3") |>
groupby(_,[:population, :location]) |>
DataFrames.combine(_, :value => (x -> [mean(x) median(x) minimum(x) maximum(x)] )=> [:mean, :median, :min, :max]) |>
select(_, Not([:min, :max])) |>
DataFrames.stack(_, [:mean, :median]) |>
DataFrames.unstack(_, :variable, :location, :value) |>
select(_,:variable, AsTable([:bm, :blood]) => ByRow(x -> x.bm/x.blood) => :value) |>
insertcols!(_, :parameter => "RDC3", :method => "2") |>
rename(_, :variable => :summary) |>
vcat(df_ratio_approaches_combined, _) |>
transform(_, :method => (x -> categorical(x, levels = ["1a","1b","1c","2"])), renamecols=false) |>
rename(_, :method => :approach)

df_ratio_approaches_combined = @pipe df_ratio_approaches_combined |>
transform(_, :parameter => rename_ratios => :parameter)

df_ratio_approaches_combined_wdc3 = @pipe df_ratio_approaches_combined_wdc3 |>
transform(_, :parameter => rename_ratios => :parameter)

end



# ╔═╡ 2574d910-d378-464c-84ee-dc2578553d83
df_ratio_approaches_combined_wdc3

# ╔═╡ dc9fdbac-76b3-11eb-319d-e9e7ef5f5e4a
md"## Analysis of cell cycle status data"

# ╔═╡ fcb04108-7754-11eb-1fc5-b1ea555f4189
fig_sg2m

# ╔═╡ cba1829c-7755-11eb-0555-eb04c8a7d671
md"As previously stated the fraction of Ki67+ cells in the blood compartment is neglible which is reflected in the model assumption that cells in the blood do not proliferate."

# ╔═╡ 3759cf24-7758-11eb-3f6a-ddf250334c6f
md"Based on the minimum, maximum and mean length of **$(mintime=5.0) hrs**, **$(meantime=10.0) hrs** and **$(maxtime=15.0) hrs** a cell spends in the the SG2M phase observed in mammalian cells we calculate a rough estimate of the proliferation rate for each populatiion based on the fraction of cell in SG2M phase is calculated as follows:"

# ╔═╡ af0221a8-7757-11eb-00ea-21622ce41781
md"**proliferation rate = (fraction\_SG2M / phase\_length) * 24h**"

# ╔═╡ 0f6a34ae-7c49-11eb-0701-079184b4cc45
md"#### Bootstraping to determine creadible priors for proliferation rate"

# ╔═╡ 5323669a-7c4f-11eb-14e8-cdb2539e5d7e
	df_cycle_long_bm = @linq df_cell_cycle |> where(:state .== "G2", :location .== "bm") |> transform(value= :value ./100)

# ╔═╡ f9882475-6e97-4026-9f08-9d50d74d41b8
begin
	@model truncnormal_model(x) = begin

		μ ~ Uniform(0,2) 
		σ ~ Uniform(0.0,2) 
		
		x .~ Truncated(Normal(μ,σ), 0.0, Inf)

	end

	@model lognormal_model(x) = begin

		μ ~ Uniform(-5,5) 
		σ ~ Uniform(0.0,5) 
		
		x .~ LogNormal(μ,σ)

	end
end

# ╔═╡ 2b6e2dc8-968a-42c2-ae0d-ab19440f627f
@model gamma_model(x) = begin

	α ~ Uniform(eps(),5) 
	θ ~ Uniform(eps(),5) 

	x .~ Gamma(α,θ)

end

# ╔═╡ 4105b6d8-e446-4b48-b039-04059c7bdce0
logpdf(LogNormal(), eps())

# ╔═╡ 7f41768d-db55-474b-9d3d-d972f87bb396
begin
Random.seed!(1234)
bootst_comb_ASDC = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "ASDC")) |> select(_, :value) |> subset(_, :value => (x -> x .!= 0.0)) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0
end

# ╔═╡ 0d3a3a1a-0bcf-4bdd-80db-8769728f2d58
begin
	Random.seed!(1234)
bootst_comb_cDC1 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "cDC1")) |> select(_, :value) |> subset(_, :value => (x -> x .!= 0.0)) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0
end

# ╔═╡ f9f438d7-6a4f-43a8-b3ac-50d080bbab45
begin
	Random.seed!(1234)
bootst_comb_DC2 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "DC2")) |> select(_, :value) |> subset(_, :value => (x -> x .!= 0.0)) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0
end

# ╔═╡ 075961aa-1ae1-460c-8ff3-34de54542a3e
begin
	Random.seed!(1234)
bootst_comb_DC3 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "DC3")) |> select(_, :value) |> subset(_, :value => (x -> x .!= 0.0)) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0
end

# ╔═╡ 8c1b9ce5-803b-45e4-adc2-1eeabf56a9cd
begin
	galac_prob_ASDC = Turing.optim_problem(truncnormal_model(bootst_comb_ASDC), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0], ub=[2.0,2.0])	
	Random.seed!(1234)
	res_ASDC= solve(galac_prob_ASDC.prob, BBO(), maxiters = 1e6);
	res_ASDC_GD= solve(remake(galac_prob_ASDC.prob, u0=res_ASDC.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ dc0bd7bb-293d-4163-b4df-073596852f33
begin
	galac_prob_ASDC_lognormal = Turing.optim_problem(lognormal_model(bootst_comb_ASDC), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[-5.0,0.0], ub=[5.0,5.0])	
	Random.seed!(1234)
	res_ASDC_lognormal= solve(galac_prob_ASDC_lognormal.prob, BBO(), maxiters = 1e6);
	res_ASDC_GD_lognormal= solve(remake(galac_prob_ASDC_lognormal.prob, u0=res_ASDC_lognormal.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 60075185-1cc9-4dc2-ab66-ae8c33a71008
begin
	galac_prob_ASDC_gamma = Turing.optim_problem(gamma_model(bootst_comb_ASDC), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0], ub=[5.0,5.0])
	Random.seed!(1234)
	res_ASDC_gamma= solve(galac_prob_ASDC_gamma.prob, BBO(), maxiters = 1e6);
	res_ASDC_GD_gamma= solve(remake(galac_prob_ASDC_gamma.prob, u0=res_ASDC_gamma.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 9204bc74-3c46-4acd-92cf-89e70a426076
res_ASDC_dist_opt = fit_mle(LogNormal, bootst_comb_ASDC)

# ╔═╡ 163246f1-eb81-45e7-8cc7-8b401eb03b74
res_ASDC_dist_opt_gamma = fit_mle(Gamma, bootst_comb_ASDC)

# ╔═╡ 0f36747b-3392-48ea-a829-9831a5f031e4
begin
	galac_prob_cDC1 = Turing.optim_problem(truncnormal_model(bootst_comb_cDC1), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0], ub=[2.0,2.0])	
	Random.seed!(1234)
	res_cDC1= solve(galac_prob_cDC1.prob, BBO(), maxiters = 1e6);
	res_cDC1_GD= solve(remake(galac_prob_cDC1.prob, u0=res_cDC1.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 203e018a-2487-4ffc-b356-51ba47ff3403
begin
	galac_prob_cDC1_lognormal = Turing.optim_problem(lognormal_model(bootst_comb_cDC1), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[-5.0,0.0], ub=[5.0,5.0])	
	Random.seed!(1234)
	res_cDC1_lognormal= solve(galac_prob_cDC1_lognormal.prob, BBO(), maxiters = 1e6);
	res_cDC1_GD_lognormal= solve(remake(galac_prob_cDC1_lognormal.prob, u0=res_cDC1_lognormal.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 4f603035-4d8d-40e0-8ee0-2383e6e3eb5c
begin
	galac_prob_cDC1_gamma = Turing.optim_problem(gamma_model(bootst_comb_cDC1), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0] .+eps(), ub=[5.0,5.0])	
	Random.seed!(1234)
	res_cDC1_gamma= solve(galac_prob_cDC1_gamma.prob, BBO(), maxiters = 1e5);
	res_cDC1_GD_gamma= solve(remake(galac_prob_cDC1_gamma.prob, u0=res_cDC1_gamma.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 4ec04374-5190-4358-851c-b2558eab676d
res_cDC1_dist_opt = fit_mle(LogNormal, bootst_comb_cDC1.+ eps())

# ╔═╡ 138905a3-d8db-4065-b5df-c60ee9a17438
res_cDC1_dist_opt_gamma = fit_mle(Gamma, bootst_comb_cDC1 .+ eps())

# ╔═╡ 1162d297-0776-43cf-b47c-a30d175e70eb
begin
	galac_prob_DC2 = Turing.optim_problem(truncnormal_model(bootst_comb_DC2), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0], ub=[2.0,2.0])	
	Random.seed!(1234)
	res_DC2= solve(galac_prob_DC2.prob, BBO(), maxiters = 1e6);
	res_DC2_GD= solve(remake(galac_prob_DC2.prob, u0=res_DC2.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 1ed28869-c6b9-4ec5-b84a-e11c64501d47
begin
	galac_prob_DC2_lognormal = Turing.optim_problem(lognormal_model(bootst_comb_DC2), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[-5.0,0.0], ub=[5.0,5.0])	
	Random.seed!(1234)
	res_DC2_lognormal= solve(galac_prob_DC2_lognormal.prob, BBO(), maxiters = 1e6);
	res_DC2_GD_lognormal= solve(remake(galac_prob_DC2_lognormal.prob, u0=res_DC2_lognormal.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 0c0b94bd-b9df-4f7f-905a-e26e6b7e0336
begin
	galac_prob_DC2_gamma = Turing.optim_problem(gamma_model(bootst_comb_DC2), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0] .+eps(), ub=[5.0,5.0])	
	Random.seed!(1234)
	res_DC2_gamma= solve(galac_prob_DC2_gamma.prob, BBO(), maxiters = 1e5);
	res_DC2_GD_gamma= solve(remake(galac_prob_DC2_gamma.prob, u0=res_DC2_gamma.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ c09d1291-914d-491c-9ec2-e9f1ca1f6b56
res_DC2_dist_opt = fit_mle(Normal, bootst_comb_DC2)

# ╔═╡ 26d09995-2e7b-40b5-ba24-7d1e8dcbfaea
res_DC2_dist_opt_gamma = fit_mle(Gamma, bootst_comb_DC2 .+ eps())

# ╔═╡ c6d5216b-1981-41b3-8c44-3d75f19b0a7d
begin
	galac_prob_DC3 = Turing.optim_problem(truncnormal_model(bootst_comb_DC3), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0], ub=[2.0,2.0])	
	Random.seed!(1234)
	res_DC3= solve(galac_prob_DC3.prob, BBO(), maxiters = 1e6);
	res_DC3_GD= solve(remake(galac_prob_DC3.prob, u0=res_DC3.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 6ff33721-f9a6-43c3-b1cb-a11794714151
begin
	galac_prob_DC3_lognormal = Turing.optim_problem(lognormal_model(bootst_comb_DC3), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[-5.0,0.0], ub=[5.0,5.0])	
	Random.seed!(1234)
	res_DC3_lognormal= solve(galac_prob_DC3_lognormal.prob, BBO(), maxiters = 1e6);
	res_DC3_GD_lognormal= solve(remake(galac_prob_DC3_lognormal.prob, u0=res_DC3_lognormal.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ dd8c8d81-7aa1-4c03-8037-7a88398a9971
begin
	galac_prob_DC3_gamma = Turing.optim_problem(gamma_model(bootst_comb_DC3), MLE();constrained=true, autoad = GalacticOptim.AutoForwardDiff(), lb=[0.0,0.0] .+eps(), ub=[5.0,5.0])	
	Random.seed!(1234)
	res_DC3_gamma= solve(galac_prob_DC3_gamma.prob, BBO(), maxiters = 1e5);
	res_DC3_GD_gamma= solve(remake(galac_prob_DC3_gamma.prob, u0=res_DC3_gamma.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 5c92cc2b-d677-4861-ae90-66eded3fdaab
res_DC3_dist_opt = fit_mle(Normal, bootst_comb_DC3 .+ eps())

# ╔═╡ 92f5e1ab-06a9-493f-b630-53e8327cf07a
res_DC3_dist_opt_gamma = fit_mle(Gamma, bootst_comb_DC3 .+ eps())

# ╔═╡ 91a9b444-7c71-11eb-3858-17f5c2b4c884
md"Both bootstrap and plain sampling yielded comparable results. We will be using the bootstrapping method, which in essence combines bootstrap samples from G2 fraction with samples from a uniform distribution U(5.0, 15.0). The priors of the proliferation rates used in the inference are the following:"

# ╔═╡ 41aa65d3-5367-4e2c-9a3b-041909ec49ad
df_p_priors_truncated_normal = DataFrame(
	parameter = ["ASDC","cDC1", "DC2", "DC3"],
	µ = [res_ASDC_GD.u[1],res_cDC1_GD.u[1],res_DC2_GD.u[1], res_DC3_GD_gamma.u[1]],
	σ = [res_ASDC_GD.u[2],res_cDC1_GD.u[2],res_DC2_GD.u[2], res_DC3_GD_gamma.u[2]],
	dist = "Normal",
	truncated = 1,
	lower = 0.0,
	upper = Inf
)

# ╔═╡ 0fb66180-d1b3-4c63-bdb4-3e86f69fb4d2
begin
	df_p_priors_lognormal = DataFrame(
	parameter = ["ASDC","cDC1", "DC2", "DC3"],
	µ = [res_ASDC_GD_lognormal.u[1],res_cDC1_GD_lognormal.u[1],res_DC2_GD_lognormal.u[1], res_DC3_GD_lognormal.u[1]],
	σ = [res_ASDC_GD_lognormal.u[2],res_cDC1_GD_lognormal.u[2],res_DC2_GD_lognormal.u[2], res_DC3_GD_lognormal.u[2]],
	dist = "LogNormal",
	truncated = 0,
	lower = -Inf,
	upper = Inf)
	
# 	df_p_priors_lognormal = DataFrame(
# parameter = ["ASDC","cDC1", "DC2", "DC3"],
# µ = [res_ASDC_dist_opt.μ,res_cDC1_dist_opt.μ, res_DC2_dist_opt.μ, res_DC3_dist_opt.μ],
# σ = [res_ASDC_dist_opt.σ,res_cDC1_dist_opt.σ, res_DC2_dist_opt.σ, res_DC3_dist_opt.σ],
# dist = "LogNormal",
# truncated = 0,
# lower = -Inf,
# upper = Inf)
	
end

# ╔═╡ 36d34538-defd-444a-a3e5-587248481e05
begin
	df_p_priors_Gamma = DataFrame(
	parameter = ["ASDC","cDC1", "DC2", "DC3"],
	µ = [res_ASDC_GD_gamma.u[1],res_cDC1_GD_gamma.u[1],res_DC2_GD_gamma.u[1],res_DC3_GD_gamma.u[1]],
	σ = [res_ASDC_GD_gamma.u[2],res_cDC1_GD_gamma.u[2],res_DC2_GD_gamma.u[2],res_DC3_GD_gamma.u[2]],
	dist = "Gamma",
	truncated = 0,
	lower = -Inf,
	upper = Inf)
	
	df_p_priors_Gamma = DataFrame(
	parameter = ["ASDC","cDC1", "DC2", "DC3"],
	µ = [res_ASDC_dist_opt_gamma.α, res_cDC1_dist_opt_gamma.α, res_DC2_dist_opt_gamma.α, res_DC3_dist_opt_gamma.α],
	σ = [res_ASDC_dist_opt_gamma.θ, res_cDC1_dist_opt_gamma.θ, res_DC2_dist_opt_gamma.θ, res_DC3_dist_opt_gamma.θ],
	dist = "Gamma",
	truncated = 0,
	lower = -Inf,
	upper = Inf)
	
	
end

# ╔═╡ 587d774d-6a95-4a5f-8927-d3443fc9bf5c
begin
	fig_prior_normal = Figure()
	ax_prior_normal = [Axis(fig_prior_normal[j,i], ylabel="density") for j in 1:2 for i in 1:2]

	for (idx, j) in enumerate(eachrow(df_p_priors_truncated_normal))
		ax_prior_normal[idx].title= j.parameter
		CairoMakie.plot!(ax_prior_normal[idx], Truncated(Normal(j.μ, j.σ), 0.0, Inf), label="prior",strokewidth = 2)
		CairoMakie.hist!(ax_prior_normal[idx],[bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx], normalization=:pdf, label="bootstrap sample", color=(:red,0.0), strokecolor=:red,strokewidth = 2)
		CairoMakie.xlims!(ax_prior_normal[idx], (-0.1,maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2))
	end
	
	# legend_ax = Axis(fig_prior[3,:])
	ax_prior_normal[2].ylabel=""
	ax_prior_normal[4].ylabel=""
	
	Legend(fig_prior_normal[3,:], ax_prior_normal[1],  orientation = :horizontal, tellwidth = false, tellheight = true)
	
	
	fig_prior_normal
end

# ╔═╡ 58e06554-c1ee-4223-8231-a237c7554e20
# df_p_priors = vcat(
# 	DataFrame(df_p_priors_Gamma[1,:]),
# 	df_p_priors_truncated_normal[2:3,:],
# 	DataFrame(df_p_priors_Gamma[4,:])
# )

# ╔═╡ d2165116-022b-4826-8b0e-ee495a57799c
df_p_priors =df_p_priors_lognormal

# ╔═╡ 8eaf4eae-948e-45d7-968b-14984b44089d
function create_dist(dist::AbstractString, mu::Real, sigma::Real, truncated::Real, lower=-Inf, upper=Inf)
	mu = string(mu)
	sigma = string(sigma)
	lower = string(lower)
	upper = string(upper)
	
	dist_string = dist*"("*mu*", "*sigma*")"
	if truncated == 1
		dist_string = "truncated("*dist_string*","*lower*","*upper*")"
	end
	
	return eval(Meta.parse(dist_string))
end

# ╔═╡ 0bf0e0d2-7d2c-49e0-b647-c28bae21785d
begin
	fig_prior_lognormal = Figure()
	ax_prior_lognormal = [Axis(fig_prior_lognormal[j,i], ylabel="density") for j in 1:2 for i in 1:2]

	for (idx, j) in enumerate(eachrow(df_p_priors_lognormal))
		ax_prior_lognormal[idx].title= j.parameter
		
		
		
		# CairoMakie.plot!(ax_prior_lognormal[idx], LogNormal(j.μ, j.σ), label="prior",strokewidth = 2)
		
		CairoMakie.hist!(ax_prior_lognormal[idx],[bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx], normalization=:pdf, label="bootstrap sample", color=(:red,0.0), strokecolor=:red,strokewidth = 2)
		
		tmp_prior_dist = create_dist(j.dist, j.μ, j.σ, j.truncated, j.lower,j.upper)

		
		points = collect(0:0.00001:maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2)
		
		w = pdf.(tmp_prior_dist, points)
		# CairoMakie.plot!(ax_prior_arr[idx], tmp_prior_dist, label="fitted prior",linewidth = 3, color=color_map["prior"])
		CairoMakie.lines!(ax_prior_lognormal[idx], points, w, label="fitted prior",linewidth = 2, color=:black)
		
		
		CairoMakie.xlims!(ax_prior_lognormal[idx], (-0.1,maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2))
	end
	
	# legend_ax = Axis(fig_prior[3,:])
	ax_prior_lognormal[2].ylabel=""
	ax_prior_lognormal[4].ylabel=""
	
	Legend(fig_prior_lognormal[3,:], ax_prior_lognormal[1],  orientation = :horizontal, tellwidth = false, tellheight = true)
	
	
	fig_prior_lognormal
end

# ╔═╡ 1b3b5331-f983-410e-9f11-8e486d184d70
begin
	fig_prior = Figure()
	ax_prior = [Axis(fig_prior[j,i], ylabel="density") for j in 1:2 for i in 1:2]

	for (idx, j) in enumerate(eachrow(df_p_priors))
		ax_prior[idx].title= j.parameter
		tmp_prior_dist = create_dist(j.dist, j.μ, j.σ, j.truncated, j.lower,j.upper)
		
		CairoMakie.plot!(ax_prior[idx], tmp_prior_dist, label="prior",strokewidth = 2)
		
		CairoMakie.hist!(ax_prior[idx],[bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx], normalization=:pdf, label="bootstrap sample", color=(:red,0.0), strokecolor=:red,strokewidth = 2)
		CairoMakie.xlims!(ax_prior[idx], (-0.1,maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2))
	end
	
	# legend_ax = Axis(fig_prior[3,:])
	ax_prior[2].ylabel=""
	ax_prior[4].ylabel=""
	
	Legend(fig_prior[3,:], ax_prior[1],  orientation = :horizontal, tellwidth = false, tellheight = true)
	
	
	fig_prior
end

# ╔═╡ 25cf0054-8c86-49ad-addc-d0149f5e02cb
md"### Plot and save prior distribution figures"

# ╔═╡ 2e72fba3-b157-43c6-9417-5d76ca0e4e42
begin
	f_prior_arr = [Figure(; resolution=(650,600)) for j in 1:4]
	ax_prior_arr = [Axis(f_prior_arr[j][1,1], ylabel="pdf", xlabel="proliferation rate (day⁻¹)", titlesize=22.0f0) for j in 1:4]
	
	color_map = Dict([ j => cgrad(:roma, 2, categorical = true)[idx] for (idx, j) in enumerate(["prior", "sample"])])
	
	for (idx, j) in enumerate(eachrow(df_p_priors))
		ax_prior_arr[idx].title= j.parameter
		tmp_prior_dist = create_dist(j.dist, j.μ, j.σ, j.truncated, j.lower,j.upper)
		
		CairoMakie.hist!(ax_prior_arr[idx],[bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx], normalization=:pdf, label="Estimated proliferation rate\n(Bootstrapping)",  strokecolor=(color_map["sample"], 0.75), color = (color_map["sample"], 0), strokewidth = 2)
		
		points = collect(0:0.00001:maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2)
		w = pdf.(tmp_prior_dist, points)
		# CairoMakie.plot!(ax_prior_arr[idx], tmp_prior_dist, label="fitted prior",linewidth = 3, color=color_map["prior"])
		CairoMakie.lines!(ax_prior_arr[idx], points, w, label="fitted prior\n" * String(typeof(tmp_prior_dist).name.name) * "(μ="*string(round(tmp_prior_dist.μ; digits= 4))*",σ="*string(round(tmp_prior_dist.σ; digits= 4))*")",linewidth = 3, color=color_map["prior"])
		CairoMakie.xlims!(ax_prior_arr[idx], (-0.1,maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2))
		
		axislegend(ax_prior_arr[idx])	
	end
	

	
	f_prior_arr
end

# ╔═╡ 02d40ebe-ce25-40dd-8fef-ba3889951895
f_prior_arr[1]

# ╔═╡ bc635c77-106f-4291-895b-db936e336f7e
f_prior_arr[2]

# ╔═╡ f3e0b4bd-2a52-4b26-b130-47f370ad693f
f_prior_arr[3]

# ╔═╡ b42c3a72-f166-488f-a79f-beded6ca35a6
f_prior_arr[4]

# ╔═╡ 0b4e15a2-4b77-4596-b6ef-30a7fdfb7219
begin
	mkpath(projectdir("notebooks", "01_processing", notebook_dir_name, "results"))
	for j in f_prior_arr
		DrWatson.save
		save(projectdir("notebooks", "01_processing", notebook_dir_name, "results","prior_proliferation_"* j.current_axis.x.title[]*".pdf"), j)
	end
end

# ╔═╡ 078601d7-bb0e-4178-b99c-499e0ff4162c
md"## Save ratios to hardrive"

# ╔═╡ aba96283-01c7-4450-bd5d-3e04043d2075
save(datadir("exp_pro", "cell_ratios_revision.csv"), df_ratio_approaches_combined_wdc3)

# ╔═╡ eba36176-d1cf-4e7a-b2bd-0134467365e4
save(datadir("exp_pro", "cell_ratios_revision.bson"), :df_ratios=>df_ratio_approaches_combined_wdc3)

# ╔═╡ b18056b8-2b24-11eb-3cf9-b54bd1f27a09
md"## Save priors to harddrive"

# ╔═╡ 000d9ae4-875e-11eb-01ab-41fe46093eed
md"We save the new prior parameters both as CSV and BSON files to be used in the downstream analysis and modelling"

# ╔═╡ 17de5082-7c73-11eb-2b3a-c712a6d6664e
save(datadir("exp_pro", "p_priors_revision.csv"), df_p_priors)

# ╔═╡ 172900ce-7c73-11eb-1ace-1919cfed3ac0
save(datadir("exp_pro", "p_priors_revision.bson"), :df_p_priors=>df_p_priors)

# ╔═╡ efaf5444-2b1e-11eb-1d33-19962296cb3f
md"## Dependencies"

# ╔═╡ 981b41fa-4b8b-4427-b560-f594d82dbb05
set_aog_theme!()

# ╔═╡ Cell order:
# ╠═b54b0f2e-2b1d-11eb-0334-f32dab087681
# ╠═11944318-873a-11eb-1661-f56a06d1cb4c
# ╟─e007a4a2-2b1d-11eb-3649-91130111debc
# ╟─dd13ffa2-2b1d-11eb-00db-47d4a6ad1305
# ╟─dab7bec4-2b1d-11eb-2ff4-034c27c6c5f1
# ╟─3a59091e-2b21-11eb-13e1-9707f83f67de
# ╠═d76acba8-7690-11eb-37b5-f903fb128ee0
# ╟─73e87950-2b21-11eb-22b0-f70d8354e6c0
# ╟─e6a0a11c-9eb7-4fce-bcad-14fccfc33421
# ╟─a7177ccc-2b21-11eb-26e8-dd349d40f1f0
# ╠═7ce6e982-d06f-4ac2-a8a8-3bb979e695e0
# ╟─ba837004-2b21-11eb-33f9-efd996b7f6e8
# ╠═57200dd2-09b4-4bac-b2ce-f8ccb5b0b2d3
# ╟─39e8839c-7549-11eb-0e7e-efd773f2441a
# ╟─39c9d2ba-7549-11eb-2722-c3d6c5a4a836
# ╠═7d4edb59-a104-4b66-906d-e77b4aced31b
# ╠═6bc30ed8-32fe-4a4f-a522-113dba6d5837
# ╟─a2e62d9a-7691-11eb-2a67-75febee39da9
# ╟─b859d2da-7691-11eb-3c4b-23aaf806d05c
# ╠═ec6f9ad2-7691-11eb-1f9a-1f4ea84d92d7
# ╠═0a981ce6-7697-11eb-30d4-318124f079a5
# ╠═708537f8-76b7-11eb-2cb2-d19676df1dfd
# ╠═520dbc20-fdd8-451a-80fb-8e2d3232466f
# ╟─27af2e26-7b7b-11eb-1eaa-f5cabf39942a
# ╠═80290752-7b7b-11eb-2a1d-5707c650a0b0
# ╟─8007f76e-7753-11eb-2d1e-49006c5fa6f2
# ╟─310fcb44-769c-11eb-27f9-139e551fa012
# ╟─5db2582e-774b-11eb-076b-81fab3b405fd
# ╟─6f296b4c-774b-11eb-09c1-55b81908299a
# ╟─309dd70c-7753-11eb-1777-47b2a12efaf1
# ╠═9967f735-b5e2-459a-b75c-43468af2baf1
# ╠═0436f3ba-769c-11eb-1c45-fd1dc9a8d75d
# ╠═5749840a-774c-11eb-2fb4-afef062e407f
# ╠═5edce4f2-774c-11eb-16f7-ffcfe2e6ff34
# ╠═2c7b6d3a-774c-11eb-354d-85b49dd83eeb
# ╟─24c1813f-bad7-47ba-a581-32c13ce4a451
# ╠═88ce1361-0a57-4180-8b2c-9125801c51b1
# ╠═2574d910-d378-464c-84ee-dc2578553d83
# ╟─dc9fdbac-76b3-11eb-319d-e9e7ef5f5e4a
# ╠═fcb04108-7754-11eb-1fc5-b1ea555f4189
# ╟─cba1829c-7755-11eb-0555-eb04c8a7d671
# ╟─3759cf24-7758-11eb-3f6a-ddf250334c6f
# ╟─af0221a8-7757-11eb-00ea-21622ce41781
# ╟─0f6a34ae-7c49-11eb-0701-079184b4cc45
# ╠═5323669a-7c4f-11eb-14e8-cdb2539e5d7e
# ╠═f9882475-6e97-4026-9f08-9d50d74d41b8
# ╠═2b6e2dc8-968a-42c2-ae0d-ab19440f627f
# ╠═4105b6d8-e446-4b48-b039-04059c7bdce0
# ╠═7f41768d-db55-474b-9d3d-d972f87bb396
# ╠═0d3a3a1a-0bcf-4bdd-80db-8769728f2d58
# ╠═f9f438d7-6a4f-43a8-b3ac-50d080bbab45
# ╠═075961aa-1ae1-460c-8ff3-34de54542a3e
# ╠═8c1b9ce5-803b-45e4-adc2-1eeabf56a9cd
# ╠═dc0bd7bb-293d-4163-b4df-073596852f33
# ╠═60075185-1cc9-4dc2-ab66-ae8c33a71008
# ╠═9204bc74-3c46-4acd-92cf-89e70a426076
# ╠═163246f1-eb81-45e7-8cc7-8b401eb03b74
# ╠═0f36747b-3392-48ea-a829-9831a5f031e4
# ╠═203e018a-2487-4ffc-b356-51ba47ff3403
# ╠═4f603035-4d8d-40e0-8ee0-2383e6e3eb5c
# ╠═4ec04374-5190-4358-851c-b2558eab676d
# ╠═138905a3-d8db-4065-b5df-c60ee9a17438
# ╠═1162d297-0776-43cf-b47c-a30d175e70eb
# ╠═1ed28869-c6b9-4ec5-b84a-e11c64501d47
# ╠═0c0b94bd-b9df-4f7f-905a-e26e6b7e0336
# ╠═c09d1291-914d-491c-9ec2-e9f1ca1f6b56
# ╠═26d09995-2e7b-40b5-ba24-7d1e8dcbfaea
# ╠═c6d5216b-1981-41b3-8c44-3d75f19b0a7d
# ╠═6ff33721-f9a6-43c3-b1cb-a11794714151
# ╠═dd8c8d81-7aa1-4c03-8037-7a88398a9971
# ╠═5c92cc2b-d677-4861-ae90-66eded3fdaab
# ╠═92f5e1ab-06a9-493f-b630-53e8327cf07a
# ╟─91a9b444-7c71-11eb-3858-17f5c2b4c884
# ╠═41aa65d3-5367-4e2c-9a3b-041909ec49ad
# ╠═0fb66180-d1b3-4c63-bdb4-3e86f69fb4d2
# ╠═36d34538-defd-444a-a3e5-587248481e05
# ╠═587d774d-6a95-4a5f-8927-d3443fc9bf5c
# ╠═0bf0e0d2-7d2c-49e0-b647-c28bae21785d
# ╠═58e06554-c1ee-4223-8231-a237c7554e20
# ╠═d2165116-022b-4826-8b0e-ee495a57799c
# ╠═8eaf4eae-948e-45d7-968b-14984b44089d
# ╠═1b3b5331-f983-410e-9f11-8e486d184d70
# ╠═25cf0054-8c86-49ad-addc-d0149f5e02cb
# ╠═2e72fba3-b157-43c6-9417-5d76ca0e4e42
# ╠═02d40ebe-ce25-40dd-8fef-ba3889951895
# ╠═bc635c77-106f-4291-895b-db936e336f7e
# ╠═f3e0b4bd-2a52-4b26-b130-47f370ad693f
# ╠═b42c3a72-f166-488f-a79f-beded6ca35a6
# ╠═0b4e15a2-4b77-4596-b6ef-30a7fdfb7219
# ╠═078601d7-bb0e-4178-b99c-499e0ff4162c
# ╠═aba96283-01c7-4450-bd5d-3e04043d2075
# ╠═eba36176-d1cf-4e7a-b2bd-0134467365e4
# ╠═b18056b8-2b24-11eb-3cf9-b54bd1f27a09
# ╟─000d9ae4-875e-11eb-01ab-41fe46093eed
# ╠═17de5082-7c73-11eb-2b3a-c712a6d6664e
# ╠═172900ce-7c73-11eb-1ace-1919cfed3ac0
# ╟─efaf5444-2b1e-11eb-1d33-19962296cb3f
# ╠═8ad38cac-1a4e-459e-8712-8c3a58877ff1
# ╠═e10fa7f7-0bd8-4c7a-a608-ad9092caae23
# ╠═d22ae2f4-2b1d-11eb-1a18-ff8b94d1d0c1
# ╠═981b41fa-4b8b-4427-b560-f594d82dbb05
