### A Pluto.jl notebook ###
# v0.19.27

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
	using Optimization
	using OptimizationOptimJL
	using OptimizationBBO
	using Optim
	using Pipe
	using CategoricalArrays
	using AlgebraOfGraphics
	using CairoMakie
	using Pipe
	using CSV
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
	df_cell_blood_conc = CSV.read(datadir("exp_raw","cells","cell_count_blood.csv"), DataFrame)
	
	df_cell_blood_conc = df_cell_blood_conc .* blood_vol
	df_cell_blood_conc.donor = ("donor_blood_" .* string.(collect(1:nrow(df_cell_blood_conc))))
	df_cell_blood_conc.location = repeat(["blood"], nrow(df_cell_blood_conc))
	rename!(df_cell_blood_conc, Symbol(" ASDC") => :ASDC)
	names(df_cell_blood_conc)
end

# ╔═╡ 0c10d4af-e9a6-408b-905e-f1ee266d96c8
df_cell_blood_conc_updated = @pipe datadir("exp_raw","cells","cell_count_blood_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
(_ .* blood_vol) |>
insertcols(_, :donor => "donor_blood_" .* string.(collect((nrow(df_cell_blood_conc)+1):(nrow(df_cell_blood_conc)+nrow(_))))) |>
rename!(_, Symbol("pre-DC") => :ASDC, Symbol("CD5+ DC2") => :DC2, Symbol("CD5- DC3") => :DC3) |>
insertcols(_, :location => "blood") |>
vcat(df_cell_blood_conc, _; cols=:union)

# ╔═╡ 4c87cf24-2b20-11eb-08ba-21c2e6bbde46
begin
	df_cell_bm_conc =  CSV.read(datadir("exp_raw","cells", "cell_count_bone_marrow.csv"), DataFrame)
	df_cell_bm_conc = df_cell_bm_conc .* bm_vol
	df_cell_bm_conc.donor = ("donor_bm_" .* string.(collect(1:nrow(df_cell_bm_conc))))
	df_cell_bm_conc.location = repeat(["bm"], nrow(df_cell_bm_conc))
	rename!(df_cell_bm_conc, Symbol(" ASDC") => :ASDC,)
	df_cell_bm_conc
end

# ╔═╡ a93f1bfd-3018-4807-998c-affd45cbcd84
df_cell_bm_conc_updated = @pipe datadir("exp_raw","cells","cell_count_bone_marrow_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
(_ .* bm_vol) |>
insertcols(_, :donor => "donor_bm_" .* string.(collect((nrow(df_cell_bm_conc)+1):(nrow(df_cell_bm_conc)+nrow(_))))) |>
rename!(_, Symbol("pre-DC") => :ASDC, Symbol("CD5+ DC2") => :DC2, Symbol("CD5- DC3") => :DC3) |>
insertcols(_, :location => "bm") |>
vcat(df_cell_bm_conc, _; cols=:union)

# ╔═╡ 1274e508-2b23-11eb-3cd7-89668baf2cde
md"Then, we combine both dataframes:"

# ╔═╡ 25978f50-2b23-11eb-1ff0-67b5a3fdf254
df_cell_concentration = vcat(df_cell_blood_conc,df_cell_bm_conc)

# ╔═╡ d0353dca-0561-4458-97b7-40f19a00a86e
df_cell_concentration_updated = vcat(df_cell_blood_conc_updated, df_cell_bm_conc_updated)

# ╔═╡ ba837004-2b21-11eb-33f9-efd996b7f6e8
md"### Cell cycle status"

# ╔═╡ c4699bbe-2b21-11eb-33e2-ad7b9ddeb94e
md"Again, we enter the data for each compartment separately and then combine the data:"

# ╔═╡ 8876e062-2b23-11eb-28c0-4bd61158ba67
begin
	df_cycle_blood =CSV.read(datadir("exp_raw","cycle", "cell_cycle_blood.csv"), DataFrame)
	df_cycle_blood.donor = ("donor_blood_" .* string.(collect(1:nrow(df_cycle_blood))))
	df_cycle_blood.location = repeat(["blood"], nrow(df_cycle_blood))
	df_cycle_blood
end

# ╔═╡ 10b1331e-a12f-4e30-b6c1-4da9d412367e
df_cycle_blood_updated = @pipe datadir("exp_raw","cycle", "cell_cycle_blood_revision_aug23.csv") |>
CSV.read(_, DataFrame) |>
insertcols(_, :location => "blood", :donor => "donor_blood_" .* string.(collect((nrow(df_cycle_blood)+1):(nrow(df_cycle_blood)+nrow(_))))) |>
vcat(df_cycle_blood, _; cols=:union)

# ╔═╡ 95c990b6-2b23-11eb-2b65-afb1b842767e
begin
	df_cycle_bm = CSV.read(datadir("exp_raw","cycle","cell_cycle_BM.csv"), DataFrame)
	df_cycle_bm.donor = ("donor_bm_" .* string.(collect(1:nrow(df_cycle_bm))))
	df_cycle_bm.location = repeat(["bm"], nrow(df_cycle_bm))
	df_cycle_bm
end

# ╔═╡ d33f35ad-5e43-4e7a-99cd-83a9d2375c38
df_cycle_bm_updated = @pipe datadir("exp_raw","cycle", "cell_cycle_BM_revision_aug23.csv") |> 
CSV.read(_, DataFrame) |>
insertcols(_, :location => "bm", :donor => "donor_bm_" .* string.(collect((nrow(df_cycle_bm)+1):(nrow(df_cycle_bm)+nrow(_))))) |>
vcat(df_cycle_bm, _; cols=:union)

# ╔═╡ b8de484a-7618-11eb-12bd-6787aa89366b
df_cell_cycle = vcat(df_cycle_blood, df_cycle_bm)

# ╔═╡ 74a0b803-9978-493e-b320-64581556e91e
df_cell_cycle_updated = vcat(df_cycle_blood_updated, df_cycle_bm_updated)

# ╔═╡ 2d83685e-7549-11eb-1994-3985c2a123e3
md"## Transform data into long format"

# ╔═╡ e902b2ea-7753-11eb-3ff4-b7799790a5ff
md"Cellc cyle data:"

# ╔═╡ 2d5a4352-7549-11eb-30b3-47b5d1dacab6
begin
	df_cycle_long = DataFrames.stack(df_cell_cycle, variable_name=:measurement)
	df_cycle_long = @linq df_cycle_long |> DataFrames.transform(:measurement => ByRow((x) -> match(r"(.*) ((ASDC)|(cDC1)|(DC2|pDC))", x).captures[1]) => :state, :measurement => ByRow((x) -> match(r"(.*) ((ASDC)|(cDC1)|(DC2|pDC))", x).captures[2]) => :population) |> DataFrames.select(Not(:measurement)) |> DataFrames.transform(:population => ByRow((x) -> ifelse(x == "ASDC", "ASDC", identity(x)))   => :population) |> DataFrames.transform(:state => ByRow((x) -> ifelse(x == "G2,M, S", "G2", identity(x))) => :state)
end

# ╔═╡ 38507c53-645d-42d4-a7c7-4ba24671c9bc
df_cycle_long_updated = @pipe df_cell_cycle_updated |> DataFrames.stack(_, variable_name = :measurement) |>
dropmissing(_, :value) |>
DataFrames.transform(_, :measurement => (x -> replace.(x , "ASDC" => "ASD")), renamecols=false) |>
DataFrames.transform(_, :measurement => (x -> replace.(x, "dc3" => "DC3", "DC23" => "DC3", "DC2G1" => "DC2 G1", "DC G2SM" => "DC1 G2SM", "ASDC" => "ASDC")), renamecols=false) |>
DataFrames.transform(_, :measurement => (x -> replace.(x , "ASD" => "ASDC")), renamecols=false) |>
DataFrames.transform(_, :measurement => (x -> replace.(x, "G2,M, S" => "G2SM", "g2sm" => "G2SM", "Go" => "G0")), renamecols=false) |>
DataFrames.transform(_, 
:measurement => ByRow(x -> match(r"(.*)(G1|G0|G2SM)(.*)", x).captures[2]) => :state) |>
DataFrames.transform(_, 
:measurement => ByRow(x -> match(r"(.*?)(ASDC|cDC1|DC2|pDC|ASDC|DC1|DC2|DC3)(.*)", x).captures[2]) => :population
) |>
DataFrames.transform(_, :state => ByRow((x) -> ifelse(x == "G2SM", "G2", identity(x))), renamecols=false) |>
DataFrames.transform(_, [:state, :population] .=> (x -> string.(x)), renamecols=false) |>
DataFrames.transform(_, :population => (x -> replace.(x , "cDC1" => "DC1")), renamecols=false) |>
DataFrames.transform(_, :population => (x -> replace.(x , "DC1" => "cDC1")), renamecols=false) |>
DataFrames.transform(_, :population => (x -> replace.(x , "ASDC" => "ASDC")), renamecols=false) |>
select(_, Not(:measurement)) 

# ╔═╡ f18c0722-7753-11eb-0958-81f7dcb07e6a
md"Cell number data:"

# ╔═╡ 189bd424-7555-11eb-10a9-bff781d5ef41
begin
	df_cell_concentration_long = DataFrames.stack(df_cell_concentration, variable_name=:population)
		
end

# ╔═╡ fc0bad6d-d12e-4878-ab34-26c6489a4a48
df_cell_concentration_long_updated =@pipe df_cell_concentration_updated |>
DataFrames.stack(_, variable_name = :population) |>
dropmissing(_, :value) |>
DataFrames.transform(_, :population => (x -> replace.(x , "ASDC" => "ASDC")), renamecols=false)

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
	
	
	ax_sg2m = @pipe df_cycle_long_updated |>
	subset(_, :state => (x -> x .== "G2")) |>
	DataFrames.transform(_, :population => (x -> categorical(x, levels=[unique(x)...])), renamecols=false) |>
	DataFrames.transform(_, :location => (x -> replace(x, "blood" => "blood", "bm" => "bone marrow")), renamecols=false) |>
	DataFrames.transform(_, :location => (x -> categorical(x, levels=["bone marrow", "blood"])), renamecols=false) |>
	data(_) * mapping(:population, :value, layout=:location) *(visual(BoxPlot, outliers=false)*mapping(color=:population) + visual(Scatter, color=:black)) |>
	draw!(subfig, _; axis=(aspect=1,),  palettes = (color = [colorant"#755494",colorant"#de3458" ,colorant"#4e65a3", colorant"#c8ab37ff"],))
	ax_sg2m[1].axis.ylabel = "% of subset in SG2M phase"
	ax_sg2m[1].axis.xticklabelrotation = 45
	ax_sg2m[2].axis.xticklabelrotation = 45
	fig_sg2m

end

# ╔═╡ 6bc30ed8-32fe-4a4f-a522-113dba6d5837
begin
	renamer_location_pop = renamer("blood" => "blood", "bm" => "bone marrow")
	fig_cell_number = Figure(resolution=(700,400))
	# ax = Axis(fig_sg2m[1, 1], title="Some plot")
	subfig_cell_number = fig_cell_number[1,1] #[Axis(fig_sg2m[1,j]) for j in 1:3]
	
    ax_cell_freq = @pipe df_cell_concentration_long_updated |>
	DataFrames.transform(_, :population => (x -> categorical(x, levels=[unique(_.population)...])), renamecols=false) |>
	DataFrames.transform(_, :location => (x -> replace(x, "blood" => "blood", "bm" => "bone marrow")), renamecols=false) |>
	DataFrames.transform(_, :location => (x -> categorical(x, levels=["bone marrow", "blood"])), renamecols=false) |>
	data(_) * mapping(:population, :value, layout=:location) *(visual(BoxPlot, outliers=false)*mapping(color=:population) + visual(Scatter, color=:black)) |> 
	draw!(subfig_cell_number,_; axis=(ylabel="# cells in compartment",aspect=1,), palettes = (color = [colorant"#755494",colorant"#de3458" ,colorant"#4e65a3", colorant"#c8ab37ff"],))
	ax_cell_freq[1].axis.xticklabelrotation = 45
	ax_cell_freq[2].axis.xticklabelrotation = 45
	
	fig_cell_number
end

# ╔═╡ a2e62d9a-7691-11eb-2a67-75febee39da9
md"## Analyse cell number data and calculate cell ratios"

# ╔═╡ b859d2da-7691-11eb-3c4b-23aaf806d05c
md"First, we calculate the intra-compartment ratios for each donor individually and then summarise the individual ratios."

# ╔═╡ ec6f9ad2-7691-11eb-1f9a-1f4ea84d92d7
df_ratios = @linq df_cell_concentration_long_updated |> where(:population .∉ Ref(["pDC", "DC2"])) |> groupby(:donor) |> transform(:ratio = first(:value)./:value)

# ╔═╡ 0a981ce6-7697-11eb-30d4-318124f079a5
begin
	df_ratios_intra = @linq df_ratios |> DataFrames.select(Not(:value)) |> groupby([:location, :population]) |> DataFrames.combine(:ratio => (x -> [mean(x) median(x) std(x) minimum(x) maximum(x)] )=> [:mean, :median, :sd, :min, :max])
end

# ╔═╡ 708537f8-76b7-11eb-2cb2-d19676df1dfd
begin
	RASDCcDC1b_mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array)[1]
	RASDCcDC1bm_mean = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array)[1]
	RASDDC2b_mean = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:mean) |> Array)[1]
	RASDDC2bm_mean = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:mean) |> Array)[1]
	
	RASDCcDC1b_median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array)[1]
	RASDCcDC1bm_median = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array)[1]
	RASDDC2b_median = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:median) |> Array)[1]
	RASDDC2bm_median = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:median) |> Array)[1]
	
	RASDCcDC1b_min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array)[1]
	RASDCcDC1bm_min = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array)[1]
	RASDDC2b_min = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:min) |> Array)[1]
	RASDDC2bm_min = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:min) |> Array)[1]
	
	RASDCcDC1b_max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array)[1]
	RASDCcDC1bm_max = (@linq df_ratios_intra |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array)[1]
	RASDDC2b_max = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "blood") |> select(:max) |> Array)[1]
	RASDDC2bm_max = (@linq df_ratios_intra |> where(:population .== "DC2", :location .== "bm") |> select(:max) |> Array)[1];
end

# ╔═╡ 27af2e26-7b7b-11eb-1eaa-f5cabf39942a
md"In order to identify the most reasonable population to base our cross-compartment calculation on (following section), we also determine the variability of each population in the both compartments:"

# ╔═╡ 80290752-7b7b-11eb-2a1d-5707c650a0b0
df_cell_vari = @linq df_cell_concentration_long_updated |> where(:population .∈ Ref(["ASDC", "cDC1", "DC2", "DC2", "DC3", "pDC"])) |> groupby([:location, :population]) |> DataFrames.combine(:value =>(x -> [mean(x) median(x) std(x) minimum(x) maximum(x)] )=> [:mean, :median, :sd, :min, :max])

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

**RASDC = ASDCbm/ASDCb**

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

# ╔═╡ 0436f3ba-769c-11eb-1c45-fd1dc9a8d75d
begin
	## Approach 1a
	df_tmp = @linq df_cell_concentration_long_updated |> where(:population .!= "pDC") |> groupby([:population, :location]) |> DataFrames.combine(:value => (x -> [mean(x) median(x) minimum(x) maximum(x)] )=> [:mean, :median, :min, :max])
	
	R_ASDC_mean = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	R_ASDC_median = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	R_ASDC_min = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	R_ASDC_max = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	RcDC1_mean = R_ASDC_mean * (RASDCcDC1b_mean/RASDCcDC1bm_mean)
	RcDC1_median = R_ASDC_median * (RASDCcDC1b_median/RASDCcDC1bm_median)
	RcDC1_min = R_ASDC_min * (RASDCcDC1b_min/RASDCcDC1bm_max)
	RcDC1_max = R_ASDC_max * (RASDCcDC1b_max/RASDCcDC1bm_min)
	
	RDC2_mean = R_ASDC_mean * (RASDDC2b_mean/RASDDC2bm_mean)
	RDC2_median = R_ASDC_median * (RASDDC2b_mean/RASDDC2bm_median)
	RDC2_min = R_ASDC_min * (RASDDC2b_min/RASDDC2bm_max)
	RDC2_max = R_ASDC_max * (RASDDC2b_max/RASDDC2bm_min)
	
	df_new = DataFrame(RASDC_cDC1_blood_mean = RASDCcDC1b_mean,
RASDC_cDC1_bm_mean = RASDCcDC1bm_mean,
RASDC_DC2_blood_mean = RASDDC2b_mean,
RASDC_DC2_bm_mean = RASDDC2bm_mean,
RASDC_cDC1_blood_median = RASDCcDC1b_median,
RASDC_cDC1_bm_median = RASDCcDC1bm_median,
RASDC_DC2_blood_median = RASDDC2b_median,
RASDC_DC2_bm_median = RASDDC2bm_median,
RASDC_cDC1_blood_min = RASDCcDC1b_min,
RASDC_cDC1_bm_min = RASDCcDC1bm_min,
RASDC_DC2_blood_min = RASDDC2b_min,
RASDC_DC2_bm_min = RASDDC2bm_min,
RASDC_cDC1_blood_max = RASDCcDC1b_max,
RASDC_cDC1_bm_max = RASDCcDC1bm_max,
RASDC_DC2_blood_max = RASDDC2b_max,
RASDC_DC2_bm_max = RASDDC2bm_max,
RASDC_mean = R_ASDC_mean,
RASDC_median = R_ASDC_median,
RASDC_min = R_ASDC_min,
RASDC_max = R_ASDC_max,
RcDC1_mean = RcDC1_mean,
RcDC1_median = RcDC1_median,
RcDC1_min = RcDC1_min,
RcDC1_max = RcDC1_max,
RDC2_mean = RDC2_mean,
RDC2_median = RDC2_median,
RDC2_min = RDC2_min,
RDC2_max = RDC2_max)

df_all_ratios = @linq DataFrames.stack(df_new) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 5749840a-774c-11eb-2fb4-afef062e407f
begin
	## Approach 1b
	RcDC1_mean_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RcDC1_median_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	RcDC1_min_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	RcDC1_max_1 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]

	R_ASDC_mean_1 = RcDC1_mean_1 * (RASDCcDC1bm_mean/RASDCcDC1b_mean)
	R_ASDC_median_1 = RcDC1_median_1 * (RASDCcDC1bm_median/RASDCcDC1b_median)
	R_ASDC_min_1 = RcDC1_min_1 * (RASDCcDC1bm_min/RASDCcDC1b_max)
	R_ASDC_max_1 = RcDC1_max_1 * (RASDCcDC1bm_max/RASDCcDC1b_min)
	
	RDC2_mean_1 = R_ASDC_mean_1 * (RASDDC2b_mean/RASDDC2bm_mean)
	RDC2_median_1 = R_ASDC_median_1 * (RASDDC2b_median/RASDDC2bm_median)
	RDC2_min_1 = R_ASDC_min_1 * (RASDDC2b_min/RASDDC2bm_max)
	RDC2_max_1 = R_ASDC_max_1 * (RASDDC2b_max/RASDDC2bm_min)
	df_1 = DataFrame(RASDC_cDC1_blood_mean = RASDCcDC1b_mean,
RASDC_cDC1_bm_mean = RASDCcDC1bm_mean,
RASDC_DC2_blood_mean = RASDDC2b_mean,
RASDC_DC2_bm_mean = RASDDC2bm_mean,
RASDC_cDC1_blood_median = RASDCcDC1b_median,
RASDC_cDC1_bm_median = RASDCcDC1bm_median,
RASDC_DC2_blood_median = RASDDC2b_median,
RASDC_DC2_bm_median = RASDDC2bm_median,
RASDC_cDC1_blood_min = RASDCcDC1b_min,
RASDC_cDC1_bm_min = RASDCcDC1bm_min,
RASDC_DC2_blood_min = RASDDC2b_min,
RASDC_DC2_bm_min = RASDDC2bm_min,
RASDC_cDC1_blood_max = RASDCcDC1b_max,
RASDC_cDC1_bm_max = RASDCcDC1bm_max,
RASDC_DC2_blood_max = RASDDC2b_max,
RASDC_DC2_bm_max = RASDDC2bm_max,
RASDC_mean = R_ASDC_mean_1,
RASDC_median = R_ASDC_median_1,
RASDC_min = R_ASDC_min_1,
RASDC_max = R_ASDC_max_1,
RcDC1_mean = RcDC1_mean_1,
RcDC1_median = RcDC1_median_1,
RcDC1_min = RcDC1_min_1,
RcDC1_max = RcDC1_max_1,
RDC2_mean = RDC2_mean_1,
RDC2_median = RDC2_median_1,
RDC2_min = RDC2_min_1,
RDC2_max = RDC2_max_1)

	df_all_ratios_1 = @linq DataFrames.stack(df_1) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 5edce4f2-774c-11eb-16f7-ffcfe2e6ff34
begin
	## Aproach 1c
	RDC2_mean_2 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RDC2_median_2 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	RDC2_min_2 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	RDC2_max_2 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]

	R_ASDC_mean_2 = RDC2_mean_2 * (RASDDC2bm_mean/RASDDC2b_mean)
	R_ASDC_median_2 = RDC2_median_2 * (RASDDC2bm_median/RASDDC2b_median)
	R_ASDC_min_2 = RDC2_min_2 * (RASDDC2bm_min/RASDDC2b_max)
	R_ASDC_max_2 = RDC2_max_2 * (RASDDC2bm_max/RASDDC2b_min)
	
	RcDC1_mean_2 = R_ASDC_mean_2 * (RASDCcDC1b_mean/RASDCcDC1bm_mean)
	RcDC1_median_2 = R_ASDC_median_2 * (RASDCcDC1b_median/RASDCcDC1bm_median)
	RcDC1_min_2 = R_ASDC_min_2 * (RASDCcDC1b_min/RASDCcDC1bm_max)
	RcDC1_max_2 = R_ASDC_max_2 * (RASDCcDC1b_max/RASDCcDC1bm_min)
	
	df_2 = DataFrame(RASDC_cDC1_blood_mean = RASDCcDC1b_mean,
RASDC_cDC1_bm_mean = RASDCcDC1bm_mean,
RASDC_DC2_blood_mean = RASDDC2b_mean,
RASDC_DC2_bm_mean = RASDDC2bm_mean,
RASDC_cDC1_blood_median = RASDCcDC1b_median,
RASDC_cDC1_bm_median = RASDCcDC1bm_median,
RASDC_DC2_blood_median = RASDDC2b_median,
RASDC_DC2_bm_median = RASDDC2bm_median,
RASDC_cDC1_blood_min = RASDCcDC1b_min,
RASDC_cDC1_bm_min = RASDCcDC1bm_min,
RASDC_DC2_blood_min = RASDDC2b_min,
RASDC_DC2_bm_min = RASDDC2bm_min,
RASDC_cDC1_blood_max = RASDCcDC1b_max,
RASDC_cDC1_bm_max = RASDCcDC1bm_max,
RASDC_DC2_blood_max = RASDDC2b_max,
RASDC_DC2_bm_max = RASDDC2bm_max,
RASDC_mean = R_ASDC_mean_2,
RASDC_median = R_ASDC_median_2,
RASDC_min = R_ASDC_min_2,
RASDC_max = R_ASDC_max_2,
RcDC1_mean = RcDC1_mean_2,
RcDC1_median = RcDC1_median_2,
RcDC1_min = RcDC1_min_2,
RcDC1_max = RcDC1_max_2,
RDC2_mean = RDC2_mean_2,
RDC2_median = RDC2_median_2,
RDC2_min = RDC2_min_2,
RDC2_max = RDC2_max_2)

	df_all_ratios_2 = @linq DataFrames.stack(df_2) |> DataFrames.transform(:variable => (ByRow(x->match(r"(.*)_([mean|min|max])", x).captures[1])) => :parameter, :variable => (ByRow(x->match(r"(.*)_((mean)|(median)|(min)|(max))", x).captures[2])) => :summary) |> DataFrames.select(Not(:variable))
end

# ╔═╡ 2c7b6d3a-774c-11eb-354d-85b49dd83eeb
begin
	## Approach 2
	R_ASDC_mean_3 = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	R_ASDC_median_3 = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	R_ASDC_min_3 = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	R_ASDC_max_3 = (@linq df_tmp |> where(:population .== "ASDC", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "ASDC", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	RcDC1_mean_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RcDC1_median_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	RcDC1_min_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	RcDC1_max_3 = (@linq df_tmp |> where(:population .== "cDC1", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "cDC1", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
	RDC2_mean_3 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:mean) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:mean) |> Array |> reshape(:))[1]
	RDC2_median_3 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:median) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:median) |> Array |> reshape(:))[1]
	
	RDC2_min_3 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:min) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:max) |> Array |> reshape(:))[1]
	
	 RDC2_max_3 = (@linq df_tmp |> where(:population .== "DC2", :location .== "bm") |> select(:max) |> Array |> reshape(:))[1] / (@linq df_tmp |> where(:population .== "DC2", :location .== "blood") |> select(:min) |> Array |> reshape(:))[1]
	
		df_3 = DataFrame(RASDC_cDC1_blood_mean = RASDCcDC1b_mean,
RASDC_cDC1_bm_mean = RASDCcDC1bm_mean,
RASDC_DC2_blood_mean = RASDDC2b_mean,
RASDC_DC2_bm_mean = RASDDC2bm_mean,
RASDC_cDC1_blood_median = RASDCcDC1b_median,
RASDC_cDC1_bm_median = RASDCcDC1bm_median,
RASDC_DC2_blood_median = RASDDC2b_median,
RASDC_DC2_bm_median = RASDDC2bm_median,
RASDC_cDC1_blood_min = RASDCcDC1b_min,
RASDC_cDC1_bm_min = RASDCcDC1bm_min,
RASDC_DC2_blood_min = RASDDC2b_min,
RASDC_DC2_bm_min = RASDDC2bm_min,
RASDC_cDC1_blood_max = RASDCcDC1b_max,
RASDC_cDC1_bm_max = RASDCcDC1bm_max,
RASDC_DC2_blood_max = RASDDC2b_max,
RASDC_DC2_bm_max = RASDDC2bm_max,
RASDC_mean = R_ASDC_mean_3,
RASDC_median = R_ASDC_median_3,
RASDC_min = R_ASDC_min_3,
RASDC_max = R_ASDC_max_3,
RcDC1_mean = RcDC1_mean_3,
RcDC1_median = RcDC1_median_3,
RcDC1_min = RcDC1_min_3,
RcDC1_max = RcDC1_max_3,
RDC2_mean = RDC2_mean_3,
RDC2_median = RDC2_median_3,
RDC2_min = RDC2_min_3,
RDC2_max = RDC2_max_3)

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
				"RASDC_cDC1_bm" => "R_precDC1bm",
				"RASDC_DC2_bm" => "R_ASDC2bm",
				"RASDC_cDC1_blood" => "R_precDC1b",
				"RASDC_DC2_blood" => "R_ASDC2b",
				"RDC3" => "R_DC3"]
		
		return replace(rnames, mapping...)
	end


	df_ratio_approaches_combined = @pipe vcat(DataFrames.transform(df_all_ratios, :value => (x -> [1 for j in x]) => :method),
DataFrames.transform(df_all_ratios_1, :value => (x -> [2 for j in x]) => :method),
DataFrames.transform(df_all_ratios_2, :value => (x -> [3 for j in x]) => :method),
DataFrames.transform(df_all_ratios_3, :value => (x -> [4 for j in x]) => :method)) |>
DataFrames.transform(_, :method => (x -> string.(x)), renamecols=false) |>
DataFrames.transform(_, :method => (x -> replace(x, "1"=>"1a", "2"=>"1b", "3"=>"1c", "4"=>"2")), renamecols=false) |>
DataFrames.transform(_, :method => (x -> categorical(x, levels=["1a", "1b", "1c","2"])), renamecols=false)

df_ratio_approaches_combined_wpdc = @pipe df_cell_concentration_long_updated |>
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
DataFrames.transform(_, :method => (x -> categorical(x, levels = ["1a","1b","1c","2"])), renamecols=false) |>
rename(_, :method => :approach)

df_ratio_approaches_combined = @pipe df_ratio_approaches_combined |>
DataFrames.transform(_, :parameter => rename_ratios => :parameter)

df_ratio_approaches_combined_wpdc = @pipe df_ratio_approaches_combined_wpdc |>
DataFrames.transform(_, :parameter => rename_ratios => :parameter)

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
	df_cycle_long_bm = @linq df_cycle_long_updated |> where(:state .== "G2", :location .== "bm") |> transform(:value= :value ./100)

# ╔═╡ f9882475-6e97-4026-9f08-9d50d74d41b8
@model lognormal_model(x) = begin
	μ ~ Uniform(0.0,2)
	σ ~ Uniform(0.0,2) 
	
	x .~ Truncated(Normal(μ,σ), 0, 2.0)

end

# ╔═╡ 7f41768d-db55-474b-9d3d-d972f87bb396
bootst_comb_ASDC = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "ASDC")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ 0d3a3a1a-0bcf-4bdd-80db-8769728f2d58
bootst_comb_cDC1 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "cDC1")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ f9f438d7-6a4f-43a8-b3ac-50d080bbab45
bootst_comb_DC2 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "DC2")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ 075961aa-1ae1-460c-8ff3-34de54542a3e
bootst_comb_DC3 = (sample((@pipe df_cycle_long_bm |> subset(_, :population => (x -> x .== "DC3")) |> select(_, :value) |> Array |> reshape(_,:)), 10000, replace=true)./ rand(Uniform(mintime, maxtime), 10000)) .* 24.0

# ╔═╡ 29b6cea0-679b-4a70-81ce-207ee0ed5737
rand(lognormal_model(bootst_comb_ASDC))

# ╔═╡ 8c1b9ce5-803b-45e4-adc2-1eeabf56a9cd
begin
	galac_prob_ASDC = Turing.optim_problem(lognormal_model(bootst_comb_ASDC), MLE();constrained=true, lb=[0.0, 0.0], ub=[2.0, 2.0])	
	res_ASDC= solve(galac_prob_ASDC.prob, LBFGS(), maxiters = 1e6);
	res_ASDC_GD= solve(remake(galac_prob_ASDC.prob, u0=res_ASDC.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 0f36747b-3392-48ea-a829-9831a5f031e4
begin
	galac_prob_cDC1 = Turing.optim_problem(lognormal_model(bootst_comb_cDC1), MLE();constrained=true, lb=[0.0, 0.0], ub=[2.0, 2.0])	
	res_cDC1= solve(galac_prob_cDC1.prob, LBFGS(), maxiters = 1e6);
	res_cDC1_GD= solve(remake(galac_prob_cDC1.prob, u0=res_cDC1.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 1162d297-0776-43cf-b47c-a30d175e70eb
begin
	galac_prob_DC2 = Turing.optim_problem(lognormal_model(bootst_comb_DC2), MLE();constrained=true, lb=[0.0, 0.0], ub=[2.0, 2.0])	
	res_DC2= solve(galac_prob_DC2.prob, LBFGS(), maxiters = 1e6);
	res_DC2_GD= solve(remake(galac_prob_DC2.prob, u0=res_DC2.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ c6d5216b-1981-41b3-8c44-3d75f19b0a7d
begin
	galac_prob_DC3 = Turing.optim_problem(lognormal_model(bootst_comb_DC3), MLE();constrained=true, lb=[0.0, 0.0], ub=[2.0, 2.0])	
	res_DC3= solve(galac_prob_DC3.prob, LBFGS(), maxiters = 1e7);
	res_DC3_GD= solve(remake(galac_prob_DC3.prob, u0=res_DC3.u), Fminbox(GradientDescent()), maxiters = 1e6);
end

# ╔═╡ 91a9b444-7c71-11eb-3858-17f5c2b4c884
md"Both bootstrap and plain sampling yield comparable results. We will be using the bootstrapping method, which in essence combines bootstrap samples from G2 fraction with samples from a uniform distribution U(5.0, 15.0). The priors of the proliferation rates used in the inference are the following:"

# ╔═╡ 41aa65d3-5367-4e2c-9a3b-041909ec49ad
df_p_priors_truncated = DataFrame(parameter = ["ASDC","cDC1", "DC2", "DC3"], µ = [res_ASDC_GD.u[1],res_cDC1_GD.u[1],res_DC2_GD.u[1], res_DC3_GD.u[1]], σ = [res_ASDC_GD.u[2],res_cDC1_GD.u[2],res_DC2_GD.u[2], res_DC3_GD.u[2]], dist = ["Truncated(Normal)", "Truncated(Normal)", "Truncated(Normal)", "Truncated(Normal)"])

# ╔═╡ 587d774d-6a95-4a5f-8927-d3443fc9bf5c
begin
	fig_prior = Figure()
	ax_prior = [Axis(fig_prior[j,i], ylabel="density") for j in 1:2 for i in 1:2]
	
	for (idx, j) in enumerate(eachrow(df_p_priors_truncated))
		ax_prior[idx].title= j.parameter
		CairoMakie.density!(ax_prior[idx], rand(Truncated(Normal(j.μ, j.σ), 0, 2.0), 10000), label="fitted prior",strokewidth = 2)
		CairoMakie.density!(ax_prior[idx],[bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx], label="bootstrap sample", color=(:red,0.0), strokecolor=:red,strokewidth = 2)
		# CairoMakie.xlims!(ax_prior[idx], (-0.0,0.5))
		CairoMakie.xlims!(ax_prior[idx], (-0.,maximum([bootst_comb_ASDC,bootst_comb_cDC1,bootst_comb_DC2,bootst_comb_DC3][idx])*1.2))
	end

	
	
	# legend_ax = Axis(fig_prior[3,:])
	ax_prior[2].ylabel=""
	ax_prior[4].ylabel=""
	
	Legend(fig_prior[3,:], ax_prior[1],  orientation = :horizontal, tellwidth = false, tellheight = true)
	
	
	fig_prior
end

# ╔═╡ 3a14b993-952a-47cc-b3a5-2138edc12d15
df_p_priors_truncated

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
# ╠═c5b59e2a-2b21-11eb-22d2-4b22602a0509
# ╠═0c10d4af-e9a6-408b-905e-f1ee266d96c8
# ╠═4c87cf24-2b20-11eb-08ba-21c2e6bbde46
# ╠═a93f1bfd-3018-4807-998c-affd45cbcd84
# ╟─1274e508-2b23-11eb-3cd7-89668baf2cde
# ╠═25978f50-2b23-11eb-1ff0-67b5a3fdf254
# ╠═d0353dca-0561-4458-97b7-40f19a00a86e
# ╟─ba837004-2b21-11eb-33f9-efd996b7f6e8
# ╟─c4699bbe-2b21-11eb-33e2-ad7b9ddeb94e
# ╠═8876e062-2b23-11eb-28c0-4bd61158ba67
# ╠═10b1331e-a12f-4e30-b6c1-4da9d412367e
# ╠═95c990b6-2b23-11eb-2b65-afb1b842767e
# ╠═d33f35ad-5e43-4e7a-99cd-83a9d2375c38
# ╠═b8de484a-7618-11eb-12bd-6787aa89366b
# ╠═74a0b803-9978-493e-b320-64581556e91e
# ╟─2d83685e-7549-11eb-1994-3985c2a123e3
# ╟─e902b2ea-7753-11eb-3ff4-b7799790a5ff
# ╠═2d5a4352-7549-11eb-30b3-47b5d1dacab6
# ╠═38507c53-645d-42d4-a7c7-4ba24671c9bc
# ╟─f18c0722-7753-11eb-0958-81f7dcb07e6a
# ╠═189bd424-7555-11eb-10a9-bff781d5ef41
# ╠═fc0bad6d-d12e-4878-ab34-26c6489a4a48
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
# ╠═29b6cea0-679b-4a70-81ce-207ee0ed5737
# ╠═8c1b9ce5-803b-45e4-adc2-1eeabf56a9cd
# ╠═0f36747b-3392-48ea-a829-9831a5f031e4
# ╠═1162d297-0776-43cf-b47c-a30d175e70eb
# ╠═c6d5216b-1981-41b3-8c44-3d75f19b0a7d
# ╟─91a9b444-7c71-11eb-3858-17f5c2b4c884
# ╠═41aa65d3-5367-4e2c-9a3b-041909ec49ad
# ╠═587d774d-6a95-4a5f-8927-d3443fc9bf5c
# ╠═3a14b993-952a-47cc-b3a5-2138edc12d15
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
