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
	using Pipe
	using CategoricalArrays
	using AlgebraOfGraphics
	using CairoMakie
	using Pipe
end

# ╔═╡ 11944318-873a-11eb-1661-f56a06d1cb4c
notebook_folder = basename(@__DIR__)

# ╔═╡ b54b0f2e-2b1d-11eb-0334-f32dab087681
md"## $(notebook_folder)"

# ╔═╡ e007a4a2-2b1d-11eb-3649-91130111debc
md"## Purpose"

# ╔═╡ dd13ffa2-2b1d-11eb-00db-47d4a6ad1305
md"""
Estimating prolifarion rate priors and intra- and inter-compartment population size ratios.
"""

# ╔═╡ dab7bec4-2b1d-11eb-2ff4-034c27c6c5f1
md"## Data"

# ╔═╡ d76acba8-7690-11eb-37b5-f903fb128ee0
raw_data_folder = datadir("exp_raw");

# ╔═╡ 3a59091e-2b21-11eb-13e1-9707f83f67de
md"The `raw` data is located here $(raw_data_folder)"

# ╔═╡ 73e87950-2b21-11eb-22b0-f70d8354e6c0
md"## Data Input"

# ╔═╡ a7177ccc-2b21-11eb-26e8-dd349d40f1f0
md"### Cell number concentration"

# ╔═╡ fd25929c-2b22-11eb-0a36-f70e823e37cb
md"First we load each dataset seperately and calculate absolute cell numbers in blood and bone marrow by multiplying the respective concentrations with $(blood_vol=5000) and $(bm_vol=1750):"

# ╔═╡ c5b59e2a-2b21-11eb-22d2-4b22602a0509
begin
	for j in 1:2
		global df_cell_blood_conc = DataFrame(load(datadir("exp_raw","cells","cell_count_blood.csv")))
	end
	df_cell_blood_conc = df_cell_blood_conc .* blood_vol
	df_cell_blood_conc.donor = ("donor_blood_" .* string.(collect(1:nrow(df_cell_blood_conc))))
	df_cell_blood_conc.location = repeat(["blood"], nrow(df_cell_blood_conc))
	rename!(df_cell_blood_conc, :PreDC => :preDC,)
	df_cell_blood_conc
end

# ╔═╡ 4c87cf24-2b20-11eb-08ba-21c2e6bbde46
begin
	df_cell_bm_conc =  DataFrame(load(datadir("exp_raw","cells", "cell_count_bone_marrow.csv")))
	df_cell_bm_conc = df_cell_bm_conc .* bm_vol
	df_cell_bm_conc.donor = ("donor_bm_" .* string.(collect(1:nrow(df_cell_bm_conc))))
	df_cell_bm_conc.location = repeat(["bm"], nrow(df_cell_bm_conc))
	rename!(df_cell_bm_conc, :PreDC => :preDC,)
	df_cell_bm_conc
end

# ╔═╡ 1274e508-2b23-11eb-3cd7-89668baf2cde
md"Then, we combine both dataframes:"

# ╔═╡ 25978f50-2b23-11eb-1ff0-67b5a3fdf254
df_cell_concentration = vcat(df_cell_blood_conc,df_cell_bm_conc)

# ╔═╡ ba837004-2b21-11eb-33f9-efd996b7f6e8
md"### Cell cycle status"

# ╔═╡ c4699bbe-2b21-11eb-33e2-ad7b9ddeb94e
md"Again, we enter the data for each compartment separately and then combine the data:"

# ╔═╡ 8876e062-2b23-11eb-28c0-4bd61158ba67
begin
	df_cycle_blood = DataFrame(load(datadir("exp_raw","cycle", "cell_cycle_blood.csv")))
	df_cycle_blood.donor = ("donor_blood_" .* string.(collect(1:nrow(df_cycle_blood))))
	df_cycle_blood.location = repeat(["blood"], nrow(df_cycle_blood))
	df_cycle_blood
end

# ╔═╡ 95c990b6-2b23-11eb-2b65-afb1b842767e
begin
	df_cycle_bm = DataFrame(load(datadir("exp_raw","cycle","cell_cycle_BM.csv")))
	df_cycle_bm.donor = ("donor_blood_" .* string.(collect(1:nrow(df_cycle_bm))))
	df_cycle_bm.location = repeat(["bm"], nrow(df_cycle_bm))
	df_cycle_bm
end

# ╔═╡ b8de484a-7618-11eb-12bd-6787aa89366b
df_cell_cycle = vcat(df_cycle_blood, df_cycle_bm)

# ╔═╡ 2d83685e-7549-11eb-1994-3985c2a123e3
md"## Transform data into long format"

# ╔═╡ e902b2ea-7753-11eb-3ff4-b7799790a5ff
md"Cellc cyle data:"

# ╔═╡ 2d5a4352-7549-11eb-30b3-47b5d1dacab6
begin
	df_cycle_long = DataFrames.stack(df_cell_cycle, variable_name=:measurement)
	df_cycle_long = @linq df_cycle_long |> DataFrames.transform(:measurement => ByRow((x) -> match(r"(.*) ((PreDC)|(cDC1)|(cDC2|pDC))", x).captures[1]) => :state, :measurement => ByRow((x) -> match(r"(.*) ((PreDC)|(cDC1)|(cDC2|pDC))", x).captures[2]) => :population) |> DataFrames.select(Not(:measurement)) |> DataFrames.transform(:population => ByRow((x) -> ifelse(x == "PreDC", "preDC", identity(x)))   => :population) |> DataFrames.transform(:state => ByRow((x) -> ifelse(x == "G2,M, S", "G2", identity(x))) => :state)
end

# ╔═╡ f18c0722-7753-11eb-0958-81f7dcb07e6a
md"Cell number data:"

# ╔═╡ 189bd424-7555-11eb-10a9-bff781d5ef41
begin
	df_cell_concentration_long = DataFrames.stack(df_cell_concentration, variable_name=:population)
		
end

# ╔═╡ 39e8839c-7549-11eb-0e7e-efd773f2441a
md"## Analyse and summarise data"

# ╔═╡ 39c9d2ba-7549-11eb-2722-c3d6c5a4a836
md"First, we visualise the cell cycle and cell number measurements in both compartments:"

# ╔═╡ 7d4edb59-a104-4b66-906d-e77b4aced31b
begin
	renamer_location = renamer("blood" => "blood", "bm" => "bone marrow")
	fig_sg2m = Figure(resolution=(700,400))
	# ax = Axis(fig_sg2m[1, 1], title="Some plot")
	subfig = fig_sg2m[1,1] #[Axis(fig_sg2m[1,j]) for j in 1:3]
	
	
ax_sg2m = @pipe df_cycle_long |>
	subset(_, :state => (x -> x .== "G2"))  |> transform(_, :population => (x -> categorical(x, levels=["preDC", "cDC1", "cDC2", "pDC"])), renamecols=false) |>
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
	
@pipe df_cell_concentration_long |>
	transform(_, :population => (x -> categorical(x, levels=["preDC", "cDC1", "cDC2", "pDC"])), renamecols=false) |>
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
df_ratios = @linq df_cell_concentration_long |> where(:population .!= "pDC") |> groupby(:donor) |> transform(ratio = first(:value)./:value)

# ╔═╡ 0a981ce6-7697-11eb-30d4-318124f079a5
begin
	df_ratios_intra = @linq df_ratios |> DataFrames.select(Not(:value)) |> groupby([:location, :population]) |> DataFrames.combine(:ratio => (x -> [mean(x) median(x) std(x) minimum(x) maximum(x)] )=> [:mean, :median, :sd, :min, :max])
end

# ╔═╡ 708537f8-76b7-11eb-2cb2-d19676df1dfd
begin
	RpreDCcDC1b_mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array)[1]
	RpreDCcDC1bm_mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array)[1]
	RpreDCcDC2b_mean = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:mean) |> Array)[1]
	RpreDCcDC2bm_mean = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:mean) |> Array)[1]
	
	RpreDCcDC1b_median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array)[1]
	RpreDCcDC1bm_median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array)[1]
	RpreDCcDC2b_median = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:median) |> Array)[1]
	RpreDCcDC2bm_median = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:median) |> Array)[1]
	
	RpreDCcDC1b_min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array)[1]
	RpreDCcDC1bm_min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array)[1]
	RpreDCcDC2b_min = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:min) |> Array)[1]
	RpreDCcDC2bm_min = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:min) |> Array)[1]
	
	RpreDCcDC1b_max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array)[1]
	RpreDCcDC1bm_max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array)[1]
	RpreDCcDC2b_max = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "blood") |> select(:max) |> Array)[1]
	RpreDCcDC2bm_max = (@linq df_ratios_intra |> where(:population .== "cDC2", :location .== "bm") |> select(:max) |> Array)[1];
end

# ╔═╡ 27af2e26-7b7b-11eb-1eaa-f5cabf39942a
md"In order to identify the most reasonable population to base our cross-compartment calculation on (following section), we also determine the variability of each population in the both compartments:"

# ╔═╡ 80290752-7b7b-11eb-2a1d-5707c650a0b0
df_cell_vari = @linq df_cell_concentration_long |> where(:population .∈ Ref(["preDC", "cDC1", "cDC2", "pDC"])) |> groupby([:location, :population]) |> DataFrames.combine(:value =>(x -> [mean(x) median(x) std(x) minimum(x) maximum(x)] )=> [:mean, :median, :sd, :min, :max])

# ╔═╡ 8007f76e-7753-11eb-2d1e-49006c5fa6f2
md"### Calculating intercompartment ratios"

# ╔═╡ 310fcb44-769c-11eb-27f9-139e551fa012
md"""
The following ratios can be directly calculated per individual and thus we can assess and incorporate the inter-individual variability straightforwardly:

**RpreDCcDC1bm** = preDCbm / cDC1bm

**RpreDCcDC2bm** = preDCbm / cDC2bm 

**RpreDCcDC1b** = preDCb / cDC1b

**RpreDCcDC2b** = preDCb / cDC2b
"""



# ╔═╡ 5db2582e-774b-11eb-076b-81fab3b405fd
md"""
Due to the lack of data from both bone marrow and blood from the same individual we use population sizes in the two compartments assessed in independent donors. This can be achieved via multiple routes. One can calculate a single accross compartment ratio and derive the remaining number (Approach 1) or one can calculate all ratios directly from the data (Approach 2). The first approach can be achieved via 3 calculation depending which accross compartment ratio one calculates first:

**Approach 1a:**

**RpreDC** = preDCbm/preDCb

**RcDC1** = RpreDC * (RpreDCcDC1b/RpreDCcDC1bm)

**RcDC2** = RpreDC * (RpreDCcDC2b/RpreDCcDC2bm)


**Approach 1b:**

**RcDC1** = cDC1bm/cDC1b

**RpreDC** = RcDC1 * RpreDCcDC1bm/RpreDCcDC1b

**RcDC2** = RpreDC * (RpreDCcDC2b/RpreDCcDC2bm)



**Approach 1c:**

**RcDC2** = cDC2Cbm/cDC2b

**RpreDC** = RcDC2 * RpreDCcDC2bm/RpreDCcDC2b

**RcDC1** = RpreDC * (RpreDCcDC1b/RpreDCcDC1bm)


**Approach 2:**

**RpreDC** = preDCbm/preDCb

**RcDC1** = cDC1bm/cDC1b

**RcDC2** = cDC2Cbm/cDC2b


"""

# ╔═╡ 6f296b4c-774b-11eb-09c1-55b81908299a
md"""
The above equations are derived as follows:

**RpreDC = preDCbm/preDCb**

RpreDC = (RpreDCcDC1bm *cDC1bm) / (RpreDCcDC1b*cDC1b)

RpreDC = cDC1bm/cDC1b * RpreDCcDC1bm/RpreDCcDC1b

**RpreDC = RcDC1 * RpreDCcDC1bm/RpreDCcDC1b**

RpreDC = (RpreDCcDC2bm *cDC2bm) / (RpreDCcDC2b*cDC2b)

RpreDC = cDC2bm/cDC2b * RpreDCcDC2bm/RpreDCcDC2b

**RpreDC = RcDC2 * RpreDCcDC2bm/RpreDCcDC2b**


-----------------------

**RcDC1 = cDC1bm/cDC1b**

RcDC1 = (preDCbm/RpreDCcDC1bm) / (preDCb/RpreDCcDC1b)

RcDC1 = (preDCbm/RpreDCcDC1bm) * (RpreDCcDC1b/preDCb)

RcDC1 = (preDCbm/preDCb) * (RpreDCcDC1b/RpreDCcDC1bm)

**RcDC1 = RpreDC * (RpreDCcDC1b/RpreDCcDC1bm)**


------------------------

**RcDC2 = cDC2Cbm/cDC2b**

**RcDC2 = RpreDC * (RpreDCcDC2b/RpreDCcDC2bm)**

"""

# ╔═╡ 309dd70c-7753-11eb-1777-47b2a12efaf1
md"Following the outlined equations above we are calculating the intercompartment ratios for all 4 approach and contrast and compare the results below:"

# ╔═╡ 0436f3ba-769c-11eb-1c45-fd1dc9a8d75d
begin
	## Approach 1a
	df_tmp = @linq df_cell_concentration_long |> where(:population .!= "pDC") |> groupby([:population, :location]) |> DataFrames.combine(:value => (x -> [mean(x) median(x) minimum(x) maximum(x)] )=> [:mean, :median, :min, :max])
	
	R_preDC_mean = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	R_preDC_median = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	R_preDC_min = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	R_preDC_max = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	RcDC1_mean = R_preDC_mean * (RpreDCcDC1b_mean/RpreDCcDC1bm_mean)
	RcDC1_median = R_preDC_median * (RpreDCcDC1b_median/RpreDCcDC1bm_median)
	RcDC1_min = R_preDC_min * (RpreDCcDC1b_min/RpreDCcDC1bm_max)
	RcDC1_max = R_preDC_max * (RpreDCcDC1b_max/RpreDCcDC1bm_min)
	
	RcDC2_mean = R_preDC_mean * (RpreDCcDC2b_mean/RpreDCcDC2bm_mean)
	RcDC2_median = R_preDC_median * (RpreDCcDC2b_mean/RpreDCcDC2bm_median)
	RcDC2_min = R_preDC_min * (RpreDCcDC2b_min/RpreDCcDC2bm_max)
	RcDC2_max = R_preDC_max * (RpreDCcDC2b_max/RpreDCcDC2bm_min)
	
	df_new = DataFrame(RpreDC_cDC1_blood_mean = RpreDCcDC1b_mean,
RpreDC_cDC1_bm_mean = RpreDCcDC1bm_mean,
RpreDC_cDC2_blood_mean = RpreDCcDC2b_mean,
RpreDC_cDC2_bm_mean = RpreDCcDC2bm_mean,
RpreDC_cDC1_blood_median = RpreDCcDC1b_median,
RpreDC_cDC1_bm_median = RpreDCcDC1bm_median,
RpreDC_cDC2_blood_median = RpreDCcDC2b_median,
RpreDC_cDC2_bm_median = RpreDCcDC2bm_median,
RpreDC_cDC1_blood_min = RpreDCcDC1b_min,
RpreDC_cDC1_bm_min = RpreDCcDC1bm_min,
RpreDC_cDC2_blood_min = RpreDCcDC2b_min,
RpreDC_cDC2_bm_min = RpreDCcDC2bm_min,
RpreDC_cDC1_blood_max = RpreDCcDC1b_max,
RpreDC_cDC1_bm_max = RpreDCcDC1bm_max,
RpreDC_cDC2_blood_max = RpreDCcDC2b_max,
RpreDC_cDC2_bm_max = RpreDCcDC2bm_max,
RpreDC_mean = R_preDC_mean,
RpreDC_median = R_preDC_median,
RpreDC_min = R_preDC_min,
RpreDC_max = R_preDC_max,
RcDC1_mean = RcDC1_mean,
RcDC1_median = RcDC1_median,
RcDC1_min = RcDC1_min,
RcDC1_max = RcDC1_max,
RcDC2_mean = RcDC2_mean,
RcDC2_median = RcDC2_median,
RcDC2_min = RcDC2_min,
RcDC2_max = RcDC2_max)

df_all_ratios = @linq DataFrames.stack(df_new) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 5749840a-774c-11eb-2fb4-afef062e407f
begin
	## Approach 1b
	RcDC1_mean_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RcDC1_median_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	RcDC1_min_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	RcDC1_max_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]

	R_preDC_mean_1 = RcDC1_mean_1 * (RpreDCcDC1bm_mean/RpreDCcDC1b_mean)
	R_preDC_median_1 = RcDC1_median_1 * (RpreDCcDC1bm_median/RpreDCcDC1b_median)
	R_preDC_min_1 = RcDC1_min_1 * (RpreDCcDC1bm_min/RpreDCcDC1b_max)
	R_preDC_max_1 = RcDC1_max_1 * (RpreDCcDC1bm_max/RpreDCcDC1b_min)
	
	RcDC2_mean_1 = R_preDC_mean_1 * (RpreDCcDC2b_mean/RpreDCcDC2bm_mean)
	RcDC2_median_1 = R_preDC_median_1 * (RpreDCcDC2b_median/RpreDCcDC2bm_median)
	RcDC2_min_1 = R_preDC_min_1 * (RpreDCcDC2b_min/RpreDCcDC2bm_max)
	RcDC2_max_1 = R_preDC_max_1 * (RpreDCcDC2b_max/RpreDCcDC2bm_min)
	df_1 = DataFrame(RpreDC_cDC1_blood_mean = RpreDCcDC1b_mean,
RpreDC_cDC1_bm_mean = RpreDCcDC1bm_mean,
RpreDC_cDC2_blood_mean = RpreDCcDC2b_mean,
RpreDC_cDC2_bm_mean = RpreDCcDC2bm_mean,
RpreDC_cDC1_blood_median = RpreDCcDC1b_median,
RpreDC_cDC1_bm_median = RpreDCcDC1bm_median,
RpreDC_cDC2_blood_median = RpreDCcDC2b_median,
RpreDC_cDC2_bm_median = RpreDCcDC2bm_median,
RpreDC_cDC1_blood_min = RpreDCcDC1b_min,
RpreDC_cDC1_bm_min = RpreDCcDC1bm_min,
RpreDC_cDC2_blood_min = RpreDCcDC2b_min,
RpreDC_cDC2_bm_min = RpreDCcDC2bm_min,
RpreDC_cDC1_blood_max = RpreDCcDC1b_max,
RpreDC_cDC1_bm_max = RpreDCcDC1bm_max,
RpreDC_cDC2_blood_max = RpreDCcDC2b_max,
RpreDC_cDC2_bm_max = RpreDCcDC2bm_max,
RpreDC_mean = R_preDC_mean_1,
RpreDC_median = R_preDC_median_1,
RpreDC_min = R_preDC_min_1,
RpreDC_max = R_preDC_max_1,
RcDC1_mean = RcDC1_mean_1,
RcDC1_median = RcDC1_median_1,
RcDC1_min = RcDC1_min_1,
RcDC1_max = RcDC1_max_1,
RcDC2_mean = RcDC2_mean_1,
RcDC2_median = RcDC2_median_1,
RcDC2_min = RcDC2_min_1,
RcDC2_max = RcDC2_max_1)

	df_all_ratios_1 = @linq DataFrames.stack(df_1) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 5edce4f2-774c-11eb-16f7-ffcfe2e6ff34
begin
	## Aproach 1c
	RcDC2_mean_2 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RcDC2_median_2 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	RcDC2_min_2 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	RcDC2_max_2 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]

	R_preDC_mean_2 = RcDC2_mean_2 * (RpreDCcDC2bm_mean/RpreDCcDC2b_mean)
	R_preDC_median_2 = RcDC2_median_2 * (RpreDCcDC2bm_median/RpreDCcDC2b_median)
	R_preDC_min_2 = RcDC2_min_2 * (RpreDCcDC2bm_min/RpreDCcDC2b_max)
	R_preDC_max_2 = RcDC2_max_2 * (RpreDCcDC2bm_max/RpreDCcDC2b_min)
	
	RcDC1_mean_2 = R_preDC_mean_2 * (RpreDCcDC1b_mean/RpreDCcDC1bm_mean)
	RcDC1_median_2 = R_preDC_median_2 * (RpreDCcDC1b_median/RpreDCcDC1bm_median)
	RcDC1_min_2 = R_preDC_min_2 * (RpreDCcDC1b_min/RpreDCcDC1bm_max)
	RcDC1_max_2 = R_preDC_max_2 * (RpreDCcDC1b_max/RpreDCcDC1bm_min)
	
	df_2 = DataFrame(RpreDC_cDC1_blood_mean = RpreDCcDC1b_mean,
RpreDC_cDC1_bm_mean = RpreDCcDC1bm_mean,
RpreDC_cDC2_blood_mean = RpreDCcDC2b_mean,
RpreDC_cDC2_bm_mean = RpreDCcDC2bm_mean,
RpreDC_cDC1_blood_median = RpreDCcDC1b_median,
RpreDC_cDC1_bm_median = RpreDCcDC1bm_median,
RpreDC_cDC2_blood_median = RpreDCcDC2b_median,
RpreDC_cDC2_bm_median = RpreDCcDC2bm_median,
RpreDC_cDC1_blood_min = RpreDCcDC1b_min,
RpreDC_cDC1_bm_min = RpreDCcDC1bm_min,
RpreDC_cDC2_blood_min = RpreDCcDC2b_min,
RpreDC_cDC2_bm_min = RpreDCcDC2bm_min,
RpreDC_cDC1_blood_max = RpreDCcDC1b_max,
RpreDC_cDC1_bm_max = RpreDCcDC1bm_max,
RpreDC_cDC2_blood_max = RpreDCcDC2b_max,
RpreDC_cDC2_bm_max = RpreDCcDC2bm_max,
RpreDC_mean = R_preDC_mean_2,
RpreDC_median = R_preDC_median_2,
RpreDC_min = R_preDC_min_2,
RpreDC_max = R_preDC_max_2,
RcDC1_mean = RcDC1_mean_2,
RcDC1_median = RcDC1_median_2,
RcDC1_min = RcDC1_min_2,
RcDC1_max = RcDC1_max_2,
RcDC2_mean = RcDC2_mean_2,
RcDC2_median = RcDC2_median_2,
RcDC2_min = RcDC2_min_2,
RcDC2_max = RcDC2_max_2)

	df_all_ratios_2 = @linq DataFrames.stack(df_2) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 2c7b6d3a-774c-11eb-354d-85b49dd83eeb
begin
	## Approach 2
	R_preDC_mean_3 = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	R_preDC_median_3 = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	R_preDC_min_3 = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	R_preDC_max_3 = (@linq df_tmp |> where(:population .== "preDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "preDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	RcDC1_mean_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RcDC1_median_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	RcDC1_min_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	RcDC1_max_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	RcDC2_mean_3 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RcDC2_median_3 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	
	RcDC2_min_3 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	
	 RcDC2_max_3 = (@linq df_tmp |> where(:population .== "cDC2", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC2", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
		df_3 = DataFrame(RpreDC_cDC1_blood_mean = RpreDCcDC1b_mean,
RpreDC_cDC1_bm_mean = RpreDCcDC1bm_mean,
RpreDC_cDC2_blood_mean = RpreDCcDC2b_mean,
RpreDC_cDC2_bm_mean = RpreDCcDC2bm_mean,
RpreDC_cDC1_blood_median = RpreDCcDC1b_median,
RpreDC_cDC1_bm_median = RpreDCcDC1bm_median,
RpreDC_cDC2_blood_median = RpreDCcDC2b_median,
RpreDC_cDC2_bm_median = RpreDCcDC2bm_median,
RpreDC_cDC1_blood_min = RpreDCcDC1b_min,
RpreDC_cDC1_bm_min = RpreDCcDC1bm_min,
RpreDC_cDC2_blood_min = RpreDCcDC2b_min,
RpreDC_cDC2_bm_min = RpreDCcDC2bm_min,
RpreDC_cDC1_blood_max = RpreDCcDC1b_max,
RpreDC_cDC1_bm_max = RpreDCcDC1bm_max,
RpreDC_cDC2_blood_max = RpreDCcDC2b_max,
RpreDC_cDC2_bm_max = RpreDCcDC2bm_max,
RpreDC_mean = R_preDC_mean_3,
RpreDC_median = R_preDC_median_3,
RpreDC_min = R_preDC_min_3,
RpreDC_max = R_preDC_max_3,
RcDC1_mean = RcDC1_mean_3,
RcDC1_median = RcDC1_median_3,
RcDC1_min = RcDC1_min_3,
RcDC1_max = RcDC1_max_3,
RcDC2_mean = RcDC2_mean_3,
RcDC2_median = RcDC2_median_3,
RcDC2_min = RcDC2_min_3,
RcDC2_max = RcDC2_max_3)

	df_all_ratios_3 = @linq DataFrames.stack(df_3) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 24c1813f-bad7-47ba-a581-32c13ce4a451
md"### Combine all results"

# ╔═╡ 88ce1361-0a57-4180-8b2c-9125801c51b1
begin
	function rename_ratios(rnames)
		mapping = ["RpreDC" => "R_preDC",
				"RcDC1" => "R_cDC1",
				"RcDC2" => "R_cDC2",
				"RpreDC_cDC1_bm" => "R_precDC1bm",
				"RpreDC_cDC2_bm" => "R_precDC2bm",
				"RpreDC_cDC1_blood" => "R_precDC1b",
				"RpreDC_cDC2_blood" => "R_precDC2b",
				"RpDC" => "R_pDC"]
		
		return replace(rnames, mapping...)
	end


	df_ratio_approaches_combined = @pipe vcat(transform(df_all_ratios, :value => (x -> [1 for j in x]) => :method),
transform(df_all_ratios_1, :value => (x -> [2 for j in x]) => :method),
transform(df_all_ratios_2, :value => (x -> [3 for j in x]) => :method),
transform(df_all_ratios_3, :value => (x -> [4 for j in x]) => :method)) |>
transform(_, :method => (x -> string.(x)), renamecols=false) |>
transform(_, :method => (x -> replace(x, "1"=>"1a", "2"=>"1b", "3"=>"1c", "4"=>"2")), renamecols=false) |>
transform(_, :method => (x -> categorical(x, levels=["1a", "1b", "1c","2"])), renamecols=false)

df_ratio_approaches_combined_wpdc = @pipe df_cell_concentration_long |>
subset(_, :population => x -> x .== "pDC") |>
groupby(_,[:population, :location]) |>
DataFrames.combine(_, :value => (x -> [mean(x) median(x) minimum(x) maximum(x)] )=> [:mean, :median, :min, :max]) |>
select(_, Not([:min, :max])) |>
DataFrames.stack(_, [:mean, :median]) |>
DataFrames.unstack(_, :variable, :location, :value) |>
select(_,:variable, AsTable([:bm, :blood]) => ByRow(x -> x.bm/x.blood) => :value) |>
insertcols!(_, :parameter => "RpDC", :method => "2") |>
rename(_, :variable => :summary) |>
vcat(df_ratio_approaches_combined, _) |>
transform(_, :method => (x -> categorical(x, levels = ["1a","1b","1c","2"])), renamecols=false) |>
rename(_, :method => :approach)

df_ratio_approaches_combined = @pipe df_ratio_approaches_combined |>
transform(_, :parameter => rename_ratios => :parameter)

df_ratio_approaches_combined_wpdc = @pipe df_ratio_approaches_combined_wpdc |>
transform(_, :parameter => rename_ratios => :parameter)

end



# ╔═╡ 2574d910-d378-464c-84ee-dc2578553d83
df_ratio_approaches_combined_wpdc

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
	df_cycle_long_bm = @linq df_cycle_long |> where(:state .== "G2", :location .== "bm") |> transform(value= :value ./100)

# ╔═╡ f9882475-6e97-4026-9f08-9d50d74d41b8
@model lognormal_model(x) = begin
	upper = 2.0
	
	μ ~ Uniform(-10,10) 
	σ ~ Uniform(0.0,10) 
	
	x .~ Truncated(LogNormal(μ,σ), 0.0, 2.0)

end

# ╔═╡ 7f41768d-db55-474b-9d3d-d972f87bb396
bootst_comb_preDC = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "preDC")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ 0d3a3a1a-0bcf-4bdd-80db-8769728f2d58
bootst_comb_cDC1 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "cDC1")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ f9f438d7-6a4f-43a8-b3ac-50d080bbab45
bootst_comb_cDC2 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "cDC2")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ 075961aa-1ae1-460c-8ff3-34de54542a3e
bootst_comb_pDC = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "pDC")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ 8c1b9ce5-803b-45e4-adc2-1eeabf56a9cd
begin
	galac_prob_preDC = Turing.optim_problem(lognormal_model(bootst_comb_preDC), MLE();constrained=true, lb=[-10.0,0.0], ub=[10.0,10.0])	
	res_preDC= solve(galac_prob_preDC.prob, ParticleSwarm(n_particles=400, lower=galac_prob_preDC.prob.lb,upper=galac_prob_preDC.prob.ub));
	res_preDC_GD= solve(remake(galac_prob_preDC.prob, u0=res_preDC.u), Fminbox(GradientDescent()));
end

# ╔═╡ 0f36747b-3392-48ea-a829-9831a5f031e4
begin
	galac_prob_cDC1 = Turing.optim_problem(lognormal_model(bootst_comb_cDC1), MLE();constrained=true, lb=[-10.0,0.0], ub=[10.0,10.0])	
	res_cDC1= solve(galac_prob_cDC1.prob, ParticleSwarm(n_particles=400, lower=galac_prob_cDC1.prob.lb,upper=galac_prob_cDC1.prob.ub));
	res_cDC1_GD= solve(remake(galac_prob_cDC1.prob, u0=res_cDC1.u), Fminbox(GradientDescent()));
end

# ╔═╡ 1162d297-0776-43cf-b47c-a30d175e70eb
begin
	galac_prob_cDC2 = Turing.optim_problem(lognormal_model(bootst_comb_cDC2), MLE();constrained=true, lb=[-10.0,0.0], ub=[10.0,10.0])	
	res_cDC2= solve(galac_prob_cDC2.prob, ParticleSwarm(n_particles=400, lower=galac_prob_cDC2.prob.lb,upper=galac_prob_cDC2.prob.ub));
	res_cDC2_GD= solve(remake(galac_prob_cDC2.prob, u0=res_cDC2.u), Fminbox(GradientDescent()));
end

# ╔═╡ c6d5216b-1981-41b3-8c44-3d75f19b0a7d
begin
	galac_prob_pDC = Turing.optim_problem(lognormal_model(bootst_comb_pDC), MLE();constrained=true, lb=[-10.0,0.0], ub=[10.0,10.0])	
	res_pDC= solve(galac_prob_pDC.prob, ParticleSwarm(n_particles=400, lower=galac_prob_pDC.prob.lb,upper=galac_prob_pDC.prob.ub));
	res_pDC_GD= solve(remake(galac_prob_pDC.prob, u0=res_pDC.u), Fminbox(GradientDescent()));
end

# ╔═╡ 91a9b444-7c71-11eb-3858-17f5c2b4c884
md"Both bootstrap and plain sampling yield comparable results. We will be using the bootstrapping method, which in essence combines bootstrap samples from G2 fraction with samples from a uniform distribution U(5.0, 15.0). The priors of the proliferation rates used in the inference are the following:"

# ╔═╡ 41aa65d3-5367-4e2c-9a3b-041909ec49ad
df_p_priors_truncated = DataFrame(parameter = ["preDC","cDC1", "cDC2", "pDC"], µ = [res_preDC_GD.u[1],res_cDC1_GD.u[1],res_cDC2_GD.u[1], res_pDC_GD.u[1]], σ = [res_preDC_GD.u[2],res_cDC1_GD.u[2],res_cDC2_GD.u[2], res_pDC_GD.u[2]], dist = ["Truncated(LogNormal)", "Truncated(LogNormal)", "Truncated(LogNormal)", "Truncated(LogNormal)"])

# ╔═╡ 587d774d-6a95-4a5f-8927-d3443fc9bf5c
begin
	fig_prior = Figure()
	ax_prior = [Axis(fig_prior[j,i], ylabel="density") for j in 1:2 for i in 1:2]

	for (idx, j) in enumerate(eachrow(df_p_priors_truncated))
		ax_prior[idx].title= j.parameter
		CairoMakie.plot!(ax_prior[idx], Truncated(LogNormal(j.μ, j.σ), 0.0, 2.0), label="prior",strokewidth = 2)
		CairoMakie.density!(ax_prior[idx],[bootst_comb_preDC,bootst_comb_cDC1,bootst_comb_cDC2,bootst_comb_pDC][idx], label="bootstrap sample", color=(:red,0.0), strokecolor=:red,strokewidth = 2)
		CairoMakie.xlims!(ax_prior[idx], (-0.1,maximum([bootst_comb_preDC,bootst_comb_cDC1,bootst_comb_cDC2,bootst_comb_pDC][idx])*1.2))
	end
	
	# legend_ax = Axis(fig_prior[3,:])
	ax_prior[2].ylabel=""
	ax_prior[4].ylabel=""
	
	Legend(fig_prior[3,:], ax_prior[1],  orientation = :horizontal, tellwidth = false, tellheight = true)
	
	
	fig_prior
end

# ╔═╡ 078601d7-bb0e-4178-b99c-499e0ff4162c
md"## Save ratios to hardrive"

# ╔═╡ aba96283-01c7-4450-bd5d-3e04043d2075
save(datadir("exp_pro", "cell_ratios.csv"), df_ratio_approaches_combined_wpdc)

# ╔═╡ eba36176-d1cf-4e7a-b2bd-0134467365e4
save(datadir("exp_pro", "cell_ratios.bson"), :df_ratios=>df_ratio_approaches_combined_wpdc)

# ╔═╡ b18056b8-2b24-11eb-3cf9-b54bd1f27a09
md"## Save priors to harddrive"

# ╔═╡ 000d9ae4-875e-11eb-01ab-41fe46093eed
md"We save the new prior parameters both as CSV and BSON files to be used in the downstream analysis and modelling"

# ╔═╡ 17de5082-7c73-11eb-2b3a-c712a6d6664e
save(datadir("exp_pro", "p_priors_truncatedlognormal.csv"), df_p_priors_truncated)

# ╔═╡ 172900ce-7c73-11eb-1ace-1919cfed3ac0
save(datadir("exp_pro", "p_priors_truncatedlognormal.bson"), :df_p_priors=>df_p_priors_truncated)

# ╔═╡ efaf5444-2b1e-11eb-1d33-19962296cb3f
md"## Dependencies"

# ╔═╡ 981b41fa-4b8b-4427-b560-f594d82dbb05
set_aog_theme!()

# ╔═╡ Cell order:
# ╟─b54b0f2e-2b1d-11eb-0334-f32dab087681
# ╠═11944318-873a-11eb-1661-f56a06d1cb4c
# ╟─e007a4a2-2b1d-11eb-3649-91130111debc
# ╠═dd13ffa2-2b1d-11eb-00db-47d4a6ad1305
# ╟─dab7bec4-2b1d-11eb-2ff4-034c27c6c5f1
# ╟─3a59091e-2b21-11eb-13e1-9707f83f67de
# ╠═d76acba8-7690-11eb-37b5-f903fb128ee0
# ╟─73e87950-2b21-11eb-22b0-f70d8354e6c0
# ╟─a7177ccc-2b21-11eb-26e8-dd349d40f1f0
# ╟─fd25929c-2b22-11eb-0a36-f70e823e37cb
# ╟─c5b59e2a-2b21-11eb-22d2-4b22602a0509
# ╟─4c87cf24-2b20-11eb-08ba-21c2e6bbde46
# ╟─1274e508-2b23-11eb-3cd7-89668baf2cde
# ╟─25978f50-2b23-11eb-1ff0-67b5a3fdf254
# ╟─ba837004-2b21-11eb-33f9-efd996b7f6e8
# ╟─c4699bbe-2b21-11eb-33e2-ad7b9ddeb94e
# ╟─8876e062-2b23-11eb-28c0-4bd61158ba67
# ╟─95c990b6-2b23-11eb-2b65-afb1b842767e
# ╠═b8de484a-7618-11eb-12bd-6787aa89366b
# ╟─2d83685e-7549-11eb-1994-3985c2a123e3
# ╟─e902b2ea-7753-11eb-3ff4-b7799790a5ff
# ╟─2d5a4352-7549-11eb-30b3-47b5d1dacab6
# ╟─f18c0722-7753-11eb-0958-81f7dcb07e6a
# ╟─189bd424-7555-11eb-10a9-bff781d5ef41
# ╟─39e8839c-7549-11eb-0e7e-efd773f2441a
# ╟─39c9d2ba-7549-11eb-2722-c3d6c5a4a836
# ╠═7d4edb59-a104-4b66-906d-e77b4aced31b
# ╠═6bc30ed8-32fe-4a4f-a522-113dba6d5837
# ╟─a2e62d9a-7691-11eb-2a67-75febee39da9
# ╟─b859d2da-7691-11eb-3c4b-23aaf806d05c
# ╠═ec6f9ad2-7691-11eb-1f9a-1f4ea84d92d7
# ╠═0a981ce6-7697-11eb-30d4-318124f079a5
# ╠═708537f8-76b7-11eb-2cb2-d19676df1dfd
# ╟─27af2e26-7b7b-11eb-1eaa-f5cabf39942a
# ╠═80290752-7b7b-11eb-2a1d-5707c650a0b0
# ╟─8007f76e-7753-11eb-2d1e-49006c5fa6f2
# ╟─310fcb44-769c-11eb-27f9-139e551fa012
# ╟─5db2582e-774b-11eb-076b-81fab3b405fd
# ╟─6f296b4c-774b-11eb-09c1-55b81908299a
# ╟─309dd70c-7753-11eb-1777-47b2a12efaf1
# ╠═0436f3ba-769c-11eb-1c45-fd1dc9a8d75d
# ╠═5749840a-774c-11eb-2fb4-afef062e407f
# ╠═5edce4f2-774c-11eb-16f7-ffcfe2e6ff34
# ╠═2c7b6d3a-774c-11eb-354d-85b49dd83eeb
# ╠═24c1813f-bad7-47ba-a581-32c13ce4a451
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
# ╠═7f41768d-db55-474b-9d3d-d972f87bb396
# ╠═0d3a3a1a-0bcf-4bdd-80db-8769728f2d58
# ╠═f9f438d7-6a4f-43a8-b3ac-50d080bbab45
# ╠═075961aa-1ae1-460c-8ff3-34de54542a3e
# ╠═8c1b9ce5-803b-45e4-adc2-1eeabf56a9cd
# ╠═0f36747b-3392-48ea-a829-9831a5f031e4
# ╠═1162d297-0776-43cf-b47c-a30d175e70eb
# ╠═c6d5216b-1981-41b3-8c44-3d75f19b0a7d
# ╟─91a9b444-7c71-11eb-3858-17f5c2b4c884
# ╠═41aa65d3-5367-4e2c-9a3b-041909ec49ad
# ╠═587d774d-6a95-4a5f-8927-d3443fc9bf5c
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
