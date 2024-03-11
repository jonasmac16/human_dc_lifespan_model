### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ 0aef9bde-cae1-11ec-3d21-233dcc3977df
using DrWatson


# ╔═╡ 0aef9bf2-cae1-11ec-2986-0562a85655d9
@quickactivate "Model of DC Differentiation"


# ╔═╡ 0aef9c38-cae1-11ec-3f1f-433070fddad4
begin
    using DifferentialEquations
    using Turing
    using DataFrames
    using DataFramesMeta
    using GalacticOptim
    using Optim
    using StatsPlots
    using CSV
    using Pipe
    using AlgebraOfGraphics
    using CairoMakie
	using BlackBoxOptim
end

# ╔═╡ 0aef9c74-cae1-11ec-3930-7d8a24b2b3d5
include(srcdir("turing_galactic.jl"))


# ╔═╡ 0aef9c88-cae1-11ec-1603-21824a9cf7c8
include(projectdir("models","ode", "U_func.jl"))


# ╔═╡ 0aef9c1a-cae1-11ec-0be7-9b486378da81
folder_name = "JM_0002_Julia_Fitting_bodywater_enrichment_alternative"

# ╔═╡ 0aef9c7e-cae1-11ec-0ab1-2fe3cc83abe2
data_label = @pipe CSV.read(datadir("exp_pro", "glucose_data_revision.csv"), DataFrame) |>
_# subset(_, AsTable([:individual, :time]) => (x -> .!((x.individual .∈ Ref(["D01", "D02", "D04"])) .& (x.time .== 1/3))))

# ╔═╡ 0aef9c90-cae1-11ec-00db-45f0282ad055
frac_exp = 0.5/24.0


# ╔═╡ 0aef9c90-cae1-11ec-0f26-b338c878f29c
tau_stop = 3.5/24.0


# ╔═╡ 0aef9c9c-cae1-11ec-01c0-4948448dcaad
tp=collect(0.0:0.01:1.0)


# ╔═╡ 0aef9c9c-cae1-11ec-390c-49da614ef956
begin
	Plots.plot(tp,  U_smooth_2stp.(tp, 0.73, 19.5, frac_exp, tau_stop, frac_exp), lab="simulation")
	Plots.xlabel!("time (d)")
	Plots.ylabel!("labelled fraction of BW")
	@df unstack(data_label, :time, :individual, :enrichment) Plots.scatter!(:time, cols(2:10))
end

# ╔═╡ 0aef9cba-cae1-11ec-2303-81619c401f96
@model function model_flat_prior(data, u_func, fr_prior_upper, delta_prior_upper_prior, tau1, tau2)
    ## parameters
    fr ~ Uniform(0.0,fr_prior_upper)
    delta ~ Uniform(0.0,delta_prior_upper_prior)
    frac ~ Uniform(0.0, 3.0)
    
    σ ~ TruncatedNormal(0.0, 2.0, eps(), Inf)
	ν ~ TruncatedNormal(0.0, 2.0, eps(), Inf) #LogNormal(2.0, 1.0)
	
	
    ## produce U(t) curve
    sim_bw = u_func.(data[:,1], fr, delta, tau1, tau2, frac)

    ## calculate likelihood
    # data[:,2] ~ MvNormal(sim_bw, σ)
	data[:,2] ~ arraydist([LocationScale(j, σ, TDist(ν)) for j in sim_bw])
		
	
    return fr, delta, frac, σ, ν
end


# ╔═╡ 73cd4cf6-d711-4235-948c-96521670ae99
Plots.plot(TruncatedNormal(0.0, 2.0, eps(), Inf))

# ╔═╡ 0aef9cc2-cae1-11ec-1b30-2b0e199c662a
gdata_label = groupby(data_label, :individual)


# ╔═╡ 0aef9cd8-cae1-11ec-274b-ad76a1dcf814
begin
	res_mle_global = Array{Any,1}()
	res_mle_local = Array{Any,1}()

	for j in gdata_label 
		data_in  = @linq j |> select(:time, :enrichment) |> Array()
		tmodel = model_flat_prior(data_in, U_smooth_2stp, 1.0, 50.0, frac_exp, tau_stop)

		gprob_global = instantiate_galacticoptim_problem(tmodel, MLE(), constrained(), [0.0,0.0,0.0,eps(),eps()],[1.0,50.0,3.0,2.0,2.0])
		res_global_tmp = GalacticOptim.solve(gprob_global.prob, BBO(); maxiters=1e7);
		push!(res_mle_global, res_global_tmp)

		gprob_local = instantiate_galacticoptim_problem(tmodel, MLE(), constrained(), [0.0,0.0,0.0,eps(),eps()],[1.0,50.0,3.0,2.0,2.0]; init_vals =res_global_tmp.minimizer)
		res_local_tmp = GalacticOptim.solve(gprob_local.prob, Fminbox(LBFGS()); maxiters=1e6, allow_f_increases=false)
		push!(res_mle_local, res_local_tmp)
	end
end

# ╔═╡ 0aef9cd8-cae1-11ec-19ca-27c92d88a69b
begin
	p0 = Plots.plot(layout=length(res_mle_local));

	for j in 1:length(res_mle_local)
		Plots.plot!(p0, tp,  U_smooth_2stp.(tp, res_mle_global[j].minimizer[[1,2]]..., frac_exp, tau_stop, res_mle_global[j].minimizer[3]), lab="MLE fit (global)", subplot=j)
		Plots.plot!(p0, tp,  U_smooth_2stp.(tp, res_mle_local[j].minimizer[[1,2]]..., frac_exp, tau_stop, res_mle_local[j].minimizer[3]), lab="MLE fit (global + local)", subplot=j)
		Plots.xlabel!(p0, "time (d)",  subplot=j)
		Plots.ylabel!(p0, "labelled fraction of BW", subplot=j)
		@df gdata_label[j] Plots.scatter!(p0, :time, :enrichment, lab=unique(:individual)[1], subplot=j)
	end
	p0

end

# ╔═╡ 0aef9cf6-cae1-11ec-20a1-55d39fa0bca2
begin
	res_map_global = Array{Any,1}()
	res_map_local = Array{Any,1}()


	for j in gdata_label 
		data_in  = @linq j |> select(:time, :enrichment) |> Array()
		tmodel = model_flat_prior(data_in, U_smooth_2stp, 1.0, 50.0, frac_exp, tau_stop)

		gprob_global = instantiate_galacticoptim_problem(tmodel, MAP(), constrained(), [0.0,0.0,0.0,eps(),eps()],[1.0,50.0,3.0,2.0,2.0])
		res_global_tmp = GalacticOptim.solve(gprob_global.prob, BBO(); maxiters=4e7);
		push!(res_map_global, res_global_tmp)

		gprob_local = instantiate_galacticoptim_problem(tmodel, MAP(), constrained(), [0.0,0.0,0.0,eps(),eps()],[1.0,50.0,3.0,2.0,2.0]; init_vals =res_global_tmp.minimizer)
		res_local_tmp = GalacticOptim.solve(gprob_local.prob, Fminbox(LBFGS()); maxiters=1e6, allow_f_increases=false)
		push!(res_map_local, res_local_tmp)
	end
end

# ╔═╡ 0aef9d00-cae1-11ec-39a8-09811c80d8a3
begin
	p1 = Plots.plot(layout=length(res_map_local), size = (1000,1000));

	for j in 1:length(res_map_local)
		Plots.plot!(p1, tp,  U_smooth_2stp.(tp, res_map_global[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_map_global[j].minimizer[3]), lab="MLE fit (global)", subplot=j)
		Plots.plot!(p1, tp,  U_smooth_2stp.(tp, res_map_local[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_map_local[j].minimizer[3]), lab="MLE fit (global + local)", subplot=j)
		Plots.xlabel!(p1, "time (d)",  subplot=j)
		Plots.ylabel!(p1, "labelled fraction of BW", subplot=j)
		@df gdata_label[j] Plots.scatter!(p1, :time, :enrichment, lab=unique(:individual)[1], subplot=j)
	end
	
	p1
end

# ╔═╡ 0aef9d46-cae1-11ec-08cb-d3af433c1844
begin
	df_ut_sol = DataFrame(time=Float64[], individual=String[], enrichment=Float64[], idx_post_sample = Int[])
	df_ut_max_sol = DataFrame(time=Float64[], individual=String[], enrichment=Float64[], fit = String[])
	for j in 1:length(res_mle_local)
   		global df_ut_max_sol = vcat(df_ut_max_sol, DataFrame(
        time = tp,  
        enrichment = U_smooth_2stp.(tp, res_mle_local[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_mle_local[j].minimizer[3]),
        fit = fill("MLE", length(tp)),
        individual = fill(first(unique(gdata_label[j].individual)), length(tp))))
    	global df_ut_max_sol =  vcat(df_ut_max_sol, DataFrame(
        time = tp,  
        enrichment = U_smooth_2stp.(tp, res_map_local[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_map_local[j].minimizer[3]),
        fit = fill("MAP", length(tp)),
        individual = fill(first(unique(gdata_label[j].individual)), length(tp))))
end
end

# ╔═╡ 5c63f5d4-284d-402b-9f04-88503a9abc81
df_ut_max_sol

# ╔═╡ 0aef9d50-cae1-11ec-290d-cd6a65172b4e
begin

	axis = (width = 225, height = 225)
	layer_2 = @pipe gdata_label |> DataFrame(_) |> data(_) * mapping(:time, :enrichment, layout=:individual) *visual(Scatter, color="red")
	layer_3 = @pipe df_ut_max_sol |> data(_) * mapping(:time, :enrichment, layout=:individual, color=:fit) *visual(Lines, linewidth=2.0)

	plt_fit = draw(layer_2 + layer_3; axis=(axis..., xlabel="time (days)", ylabel="enrichment in body water (saliva)"), facet = (; linkxaxes = :none))

end
	

# ╔═╡ 05ac534f-771a-4f8b-b84d-2acfabdf41ff
res_mle_global[8].u

# ╔═╡ 9cb09b63-396c-4bd2-9bd9-b2d4425d4b1a
res_map_global[8].u

# ╔═╡ 0aef9d5a-cae1-11ec-110f-ad3f37e7b85d
begin
    mkpath(projectdir("notebooks", "01_processing", folder_name,"results"))
    save(projectdir("notebooks", "01_processing", folder_name, "results","MLE_MAP_BW_fit.pdf"), plt_fit)
end

# ╔═╡ 0aef9d6e-cae1-11ec-3673-2568b90fd439
begin
df_label_pars = DataFrame(hcat([j.minimizer[[1,2,3]] for j in res_mle_local]...),[:C66,:C67,:C68, :C52, :C53, :C55, :D01, :D02, :D04])
df_label_pars = hcat(DataFrame(:parameter => ["fr","delta","frac"]), df_label_pars)
df_label_pars = DataFrames.unstack(DataFrames.stack(df_label_pars, Not(:parameter)),:variable, :parameter, :value)

end

# ╔═╡ 0aef9d78-cae1-11ec-28b0-5f8679a5704f
begin
	df_label_pars_map = DataFrame(hcat([j.minimizer[[1,2,3]] for j in res_map_local]...),[:C66,:C67,:C68, :C52, :C53, :C55, :D01, :D02, :D04])
	df_label_pars_map = hcat(DataFrame(:parameter => ["fr","delta","frac"]), df_label_pars_map)
	df_label_pars_map = DataFrames.unstack(DataFrames.stack(df_label_pars_map, Not(:parameter)),:variable, :parameter, :value)
end

# ╔═╡ 0aef9d82-cae1-11ec-2e1b-b3e640d5bc81
save(datadir("exp_pro", "labeling_parameters_revision.csv"), df_label_pars)


# ╔═╡ 0aef9d82-cae1-11ec-04f2-0f3fea901855
save(datadir("exp_pro", "labeling_parameters_revision.bson"),  df_label_pars => df_label_pars)


# ╔═╡ Cell order:
# ╠═0aef9bde-cae1-11ec-3d21-233dcc3977df
# ╠═0aef9bf2-cae1-11ec-2986-0562a85655d9
# ╠═0aef9c1a-cae1-11ec-0be7-9b486378da81
# ╠═0aef9c38-cae1-11ec-3f1f-433070fddad4
# ╠═0aef9c74-cae1-11ec-3930-7d8a24b2b3d5
# ╠═0aef9c7e-cae1-11ec-0ab1-2fe3cc83abe2
# ╠═0aef9c88-cae1-11ec-1603-21824a9cf7c8
# ╠═0aef9c90-cae1-11ec-00db-45f0282ad055
# ╠═0aef9c90-cae1-11ec-0f26-b338c878f29c
# ╠═0aef9c9c-cae1-11ec-01c0-4948448dcaad
# ╠═0aef9c9c-cae1-11ec-390c-49da614ef956
# ╠═0aef9cba-cae1-11ec-2303-81619c401f96
# ╠═73cd4cf6-d711-4235-948c-96521670ae99
# ╠═0aef9cc2-cae1-11ec-1b30-2b0e199c662a
# ╠═0aef9cd8-cae1-11ec-274b-ad76a1dcf814
# ╠═0aef9cd8-cae1-11ec-19ca-27c92d88a69b
# ╠═0aef9cf6-cae1-11ec-20a1-55d39fa0bca2
# ╠═0aef9d00-cae1-11ec-39a8-09811c80d8a3
# ╠═0aef9d46-cae1-11ec-08cb-d3af433c1844
# ╠═5c63f5d4-284d-402b-9f04-88503a9abc81
# ╠═0aef9d50-cae1-11ec-290d-cd6a65172b4e
# ╠═05ac534f-771a-4f8b-b84d-2acfabdf41ff
# ╠═9cb09b63-396c-4bd2-9bd9-b2d4425d4b1a
# ╠═0aef9d5a-cae1-11ec-110f-ad3f37e7b85d
# ╠═0aef9d6e-cae1-11ec-3673-2568b90fd439
# ╠═0aef9d78-cae1-11ec-28b0-5f8679a5704f
# ╠═0aef9d82-cae1-11ec-2e1b-b3e640d5bc81
# ╠═0aef9d82-cae1-11ec-04f2-0f3fea901855
