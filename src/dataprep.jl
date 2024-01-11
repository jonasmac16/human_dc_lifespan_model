function prepare_data_turing(data, data_r, data_label_p, tau; population = ["preDC", "cDC1", "DC2"], individual = ["C66", "C67", "C68", "C52", "C53", "C55"], ratios = ["R_ASDC", "R_cDC1", "R_DC2", "R_ASDCcDC1bm", "R_ASDCDC2bm", "R_ASDCcDC1b", "R_ASDCDC2b"], population_mapping =[j => idx for (idx, j) in enumerate(population)], label_p_names = [:fr,:delta, :frac], ratio_approach="2", ratio_summary = "mean", mean_data::Bool = false)
    # individual_mapping = [j => idx for (idx, j) in enumerate(individual)]
    
    df = @pipe data |> 
    subset(_, :population => x-> x .∈ Ref(population), :individual => x -> x .∈ Ref(individual)) |> 
    groupby(_, [:time, :population, :individual]) |>
    transform(_, :enrichment => std => :SD, ungroup=false) |>
    combine(_, [:enrichment,:SD] .=> (x -> mean_data ? mean(x) : identity(x))  .=> [:enrichment, :SD]) |>
    transform(_, :population => (x -> replace(x, population_mapping...)) => :population_idx) |> 
    transform(_, :individual => (x -> replace(x, [j => idx for (idx, j) in enumerate(individual[individual .∈ Ref(x)])]...)) => :individual_idx) |> 
    transform(_, [:population_idx, :individual_idx] .=> (x -> Int.(x)), renamecols=false) |>
    sort(_,[:individual_idx, :population_idx, :time]) |>
    groupby(_, [:time, :population, :individual]) |>
    hcat(parent(_), DataFrame(:technical_idx => Int.(groupindices(_)) )) |>
    groupby(_, :individual) |>
    transform(_, :time => (x ->indexin(x, sort(unique(x)))) => :timepoint_idx) |>
    sort(_, [:individual_idx, :population_idx, :time])

	n_indv = length(unique(df[:,:individual_idx]))
	t_total = Array{Array{Float64,1}, 1}(undef,n_indv)
	label_p = Array{Array{Float64,1}, 1}(undef,n_indv)

	for (k, indv) in enumerate(sort(unique(df[:,:individual_idx])))
        indv_name = @pipe df |> subset(_, :individual_idx => x -> x .== indv) |> select(_,:individual) |> unique |> Array |> first
		t_total[k] = @pipe df |> subset(_, :individual_idx => x -> x .==indv) |> _[:, :time] |> Array(_) |> unique(_) |> sort(_) 
        label_p[k] = @pipe data_label_p |> subset(_, :variable => x -> x .==indv_name) |> select(_,label_p_names) |> Array(_)[1,:] |> [_..., tau]
		# t_total[k] = sort(unique(@linq df_sub |> select(:time) |> Array |> reshape(:)))
		# label_p[k] = [(@linq data_label_p |> where(:variable .== indv) |> DataFrames.select(label_p_names...) |> Array |> reshape(:))..., tau]#(; zip(Tuple([label_p_names..., :tau]),
	end	
    R = (; zip(Tuple(Symbol.(ratios)),
    Tuple([(@pipe data_r |> subset(_, :parameter => x -> x .== j, :approach => x -> x .== ratio_approach, :summary => x -> x .== ratio_summary) |> _[:,:value] |> _[1]) for j in ratios]))...)
		
	

    return (;df = df, data = df[:,:enrichment], data_sd = df[:,:SD], metadata = (;order=(donor=df[:,:individual_idx], population=df[:,:population_idx], timepoint_idx =df[:,:timepoint_idx], technical=df[:,:technical_idx]), R=R, timepoints=t_total, label_p=label_p, n_meassurements=maximum(df[:,:technical_idx]), n_indv=n_indv))
end