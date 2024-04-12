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
end

# ╔═╡ 0aef9c74-cae1-11ec-3930-7d8a24b2b3d5
include(srcdir("turing_galactic.jl"))


# ╔═╡ 0aef9c88-cae1-11ec-1603-21824a9cf7c8
include(projectdir("models","ode", "U_func.jl"))


# ╔═╡ 0aef9c1a-cae1-11ec-0be7-9b486378da81
folder_name = "JM_0002_Julia_Fitting_bodywater_enrichment"

# ╔═╡ 0aef9c7e-cae1-11ec-0ab1-2fe3cc83abe2
data_label = CSV.read(datadir("exp_pro", "glucose_data_revision.csv"), DataFrame)


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
    frac ~ Uniform(0.0, 1.0)
    
    σ ~ Truncated(Normal(0.0, 0.1), 0.0, Inf)

    ## produce U(t) curve
    sim_bw = u_func.(data[:,1], fr, delta, tau1, tau2, frac)

    ## calculate likelihood
    data[:,2] ~ MvNormal(sim_bw, σ)

    return fr, delta, frac, σ
end


# ╔═╡ 0aef9cc2-cae1-11ec-1b30-2b0e199c662a
gdata_label = groupby(data_label, :individual)


# ╔═╡ 0aef9cce-cae1-11ec-273a-4fb94c2a8337
res_mle_global = Array{Any,1}()


# ╔═╡ 0aef9cce-cae1-11ec-10fe-279087a3e435
res_mle_local = Array{Any,1}()


# ╔═╡ 0aef9cd8-cae1-11ec-274b-ad76a1dcf814
for j in gdata_label 
    data_in  = @linq j |> select(:time, :enrichment) |> Array()
    tmodel = model_flat_prior(data_in, U_smooth_2stp, 1.0, 50.0, frac_exp, tau_stop)
    
    gprob_global = instantiate_galacticoptim_problem(tmodel, MLE(), constrained(), [0.0,0.0,0.0,0.0],[1.0,50.0,1.0,2.0])
    res_global_tmp = GalacticOptim.solve(gprob_global.prob, ParticleSwarm(;lower= [0.0,0.0,0.0,0.0], upper=[1.0,50.0,1.0,2.0], n_particles=400); maxiters=1e4);
    push!(res_mle_global, res_global_tmp)

    gprob_local = instantiate_galacticoptim_problem(tmodel, MLE(), constrained(), [0.0,0.0,0.0,0.0],[1.0,50.0,1.0,2.0]; init_vals =res_global_tmp.minimizer)
    res_local_tmp = GalacticOptim.solve(gprob_local.prob, Fminbox(LBFGS()); maxiters=1e6, allow_f_increases=false)
    push!(res_mle_local, res_local_tmp)
end


# ╔═╡ 0aef9cd8-cae1-11ec-19db-5193e044feec
p0 = Plots.plot(layout=length(res_mle_local));


# ╔═╡ 0aef9cd8-cae1-11ec-19ca-27c92d88a69b
for j in 1:length(res_mle_local)
    Plots.plot!(p0, tp,  U_smooth_2stp.(tp, res_mle_global[j].minimizer[[1,2]]..., frac_exp, tau_stop, res_mle_global[j].minimizer[3]), lab="MLE fit (global)", subplot=j)
    Plots.plot!(p0, tp,  U_smooth_2stp.(tp, res_mle_local[j].minimizer[[1,2]]..., frac_exp, tau_stop, res_mle_local[j].minimizer[3]), lab="MLE fit (global + local)", subplot=j)
    Plots.xlabel!(p0, "time (d)",  subplot=j)
    Plots.ylabel!(p0, "labelled fraction of BW", subplot=j)
    @df gdata_label[j] Plots.scatter!(p0, :time, :enrichment, lab=unique(:individual)[1], subplot=j)
end


# ╔═╡ 0aef9ce2-cae1-11ec-2e50-cd375a56e5b5
p0


# ╔═╡ 0aef9ce2-cae1-11ec-372a-07838eecfb43
res_map_global = Array{Any,1}()


# ╔═╡ 0aef9cec-cae1-11ec-3a9c-f18ed66cea9b
res_map_local = Array{Any,1}()


# ╔═╡ 0aef9cf6-cae1-11ec-20a1-55d39fa0bca2
for j in gdata_label 
    data_in  = @linq j |> select(:time, :enrichment) |> Array()
    tmodel = model_flat_prior(data_in, U_smooth_2stp, 1.0, 50.0, frac_exp, tau_stop)
    
    gprob_global = instantiate_galacticoptim_problem(tmodel, MAP(), constrained(), [0.0,0.0,0.0,0.0],[1.0,50.0,1.0,2.0])
    res_global_tmp = GalacticOptim.solve(gprob_global.prob, ParticleSwarm(;lower= [0.0,0.0,0.0,0.0], upper=[1.0,50.0,1.0,2.0], n_particles=400); maxiters=1e4);
    push!(res_map_global, res_global_tmp)

    gprob_local = instantiate_galacticoptim_problem(tmodel, MAP(), constrained(), [0.0,0.0,0.0,0.0],[1.0,50.0,1.0,2.0]; init_vals =res_global_tmp.minimizer)
    res_local_tmp = GalacticOptim.solve(gprob_local.prob, Fminbox(LBFGS()); maxiters=1e6, allow_f_increases=false)
    push!(res_map_local, res_local_tmp)
end


# ╔═╡ 0aef9cf6-cae1-11ec-2c31-85021922ee58
p1 = Plots.plot(layout=length(res_map_local));


# ╔═╡ 0aef9d00-cae1-11ec-39a8-09811c80d8a3
for j in 1:length(res_map_local)
    Plots.plot!(p1, tp,  U_smooth_2stp.(tp, res_map_global[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_map_global[j].minimizer[3]), lab="MLE fit (global)", subplot=j)
    Plots.plot!(p1, tp,  U_smooth_2stp.(tp, res_map_local[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_map_local[j].minimizer[3]), lab="MLE fit (global + local)", subplot=j)
    Plots.xlabel!(p1, "time (d)",  subplot=j)
    Plots.ylabel!(p1, "labelled fraction of BW", subplot=j)
    @df gdata_label[j] Plots.scatter!(p1, :time, :enrichment, lab=unique(:individual)[1], subplot=j)
end


# ╔═╡ a5dd9888-fc23-4480-8c5f-981a3fbc64df
p1

# ╔═╡ 0aef9d00-cae1-11ec-11f6-254c146c5757
p1


# ╔═╡ 0aef9d0a-cae1-11ec-152b-d72930ff7e95
res_posterior = Array{Any,1}()


# ╔═╡ 0aef9d14-cae1-11ec-26bc-1dbde25fb496
for j in gdata_label 
    # data_in  = @linq j |> select(:time, :enrichment) |> Array()
    # tmodel = model_flat_prior(data_in, U_smooth_2stp, 1.0, 50.0, frac_exp, tau_stop)
    
    # res_post = sample(tmodel, NUTS(4000,0.85), 8000)
    # push!(res_posterior, res_post)
end


# ╔═╡ 0aef9d14-cae1-11ec-0532-67d89a6717bd
# corner(res_posterior[1])


# ╔═╡ 0aef9d1e-cae1-11ec-25f6-d978adb8c24f
# corner(res_posterior[2])


# ╔═╡ 0aef9d28-cae1-11ec-3de9-ab5e5c6c3d6f
# corner(res_posterior[3])


# ╔═╡ 0aef9d28-cae1-11ec-2110-8db9536a141c
# corner(res_posterior[4])


# ╔═╡ 0aef9d28-cae1-11ec-32f3-23a11568a93a
# corner(res_posterior[5])


# ╔═╡ 0aef9d34-cae1-11ec-00b0-5f343f7157a9
begin
    # p0 = Plots.plot(layout=(length(res_mle_local),1)) 
    # p1 = Plots.plot(layout=(length(res_map_local),1)) 
end# corner(res_posterior[6])


# ╔═╡ 0aef9d34-cae1-11ec-22d2-552b36bab767
# p2 = plot(layout=length(res_posterior))


# ╔═╡ 0aef9d34-cae1-11ec-1b78-3d3a1cbfc31a
# for j in 1:length(res_posterior)
#     posterior_arr = hcat(Array(res_posterior[j][[:fr]]),Array(res_posterior[j][[:delta]]), Array(res_posterior[j][[:frac]]))
    
#     CSV.write(projectdir("notebooks", folder_name, "Parameter_posterior_"*first(unique(gdata_label[j].individual))*".csv"), DataFrame(posterior_arr, ["fr", "delta", "frac"]))
#     for k in sample(1:size(posterior_arr,1), 200)
#         plot!(p2, tp,  U_smooth_2stp.(tp, posterior_arr[k,[1,2]]...,frac_exp, tau_stop, posterior_arr[k,3]), lab="", subplot=j, alpha=0.05, color="blue")
#     end
#     xlabel!(p2, "time (d)",  subplot=j)
#     ylabel!(p2, "labelled fraction\nof BW", subplot=j)
#     plot!(p2, tp,  U_smooth_2stp.(tp, res_mle_local[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_mle_local[j].minimizer[3]), lab="MLE", subplot=j,alpha=0.5, color="red", w=1.5)
#     plot!(p2, tp,  U_smooth_2stp.(tp, res_map_local[j].minimizer[[1,2]]...,frac_exp, tau_stop, res_map_local[j].minimizer[3]), lab="MAP", subplot=j,alpha=0.5, color="black", w=1.5)
#     @df gdata_label[j] Plots.scatter!(p2, :time, :enrichment, lab=unique(:individual)[1], subplot=j)
# end


# ╔═╡ 0aef9d34-cae1-11ec-1f10-fd1772393856
# p2


# ╔═╡ 0aef9d3c-cae1-11ec-2598-4d9467f6864e


# ╔═╡ 0aef9d3c-cae1-11ec-3b29-492a471db6d2



# ╔═╡ 0aef9d46-cae1-11ec-08cb-d3af433c1844
begin
	df_ut_sol = DataFrame(time=Float64[], individual=String[], enrichment=Float64[], idx_post_sample = Int[])
	df_ut_max_sol = DataFrame(time=Float64[], individual=String[], enrichment=Float64[], fit = String[])
	for j in 1:length(res_mle_local)
    # for k in sample(1:size(posterior_arr,1), 200)
    #     df_ut_sol = vcat(df_ut_sol, DataFrame(
    #     time = tp,  
    #     enrichment = U_smooth_2stp.(tp, posterior_arr[k,[1,2]]...,frac_exp, tau_stop, posterior_arr[k,3]),
    #     idx_post_sample = fill(k, length(tp)),
    #     individual = fill(first(unique(gdata_label[j].individual)), length(tp))))
    # end
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

# ╔═╡ 0aef9d46-cae1-11ec-11c4-fb27e8670ea0
axis = (width = 225, height = 225)


# ╔═╡ 0aef9d50-cae1-11ec-290d-cd6a65172b4e
# layer_1 = @pipe df_ut_sol |> DataFrame(_) |> data(_) * mapping(:time, :enrichment,group=:idx_post_sample, layout=:individual) * visual(Lines, linewidth=0.1, color="grey")


# ╔═╡ 0aef9d50-cae1-11ec-1f1d-4d568c628508
layer_2 = @pipe gdata_label |> DataFrame(_) |> data(_) * mapping(:time, :enrichment, layout=:individual) *visual(Scatter, color="red")


# ╔═╡ 0aef9d50-cae1-11ec-18b5-e79e1d6e93f8
layer_3 = @pipe df_ut_max_sol |> data(_) * mapping(:time, :enrichment, layout=:individual, color=:fit) *visual(Lines, linewidth=2.0)


# ╔═╡ 0aef9d50-cae1-11ec-1d7e-0bbdbba8b961
plt_fit = draw(layer_2 + layer_3; axis=(axis..., xlabel="time (days)", ylabel="enrichment in body water (saliva)"), facet = (; linkxaxes = :none))


# ╔═╡ 0aef9d5a-cae1-11ec-110f-ad3f37e7b85d
begin
    mkpath(projectdir("notebooks", "01_processing", folder_name,"results"))
    save(projectdir("notebooks", "01_processing", folder_name, "results","MLE_MAP_BW_fit.pdf"), plt_fit)
end

# ╔═╡ 0aef9d5a-cae1-11ec-1546-5963eb87a8fd
p_dens_mle_arr = Array{Any,1}()


# ╔═╡ 0aef9d5a-cae1-11ec-2944-331eab7f0ab8
for j in 1:length(res_posterior)
    tmp_p = Plots.plot(layout=(4,1)) # density(res_posterior[j], [:fr, :delta,:frac,:σ])
    vline!(tmp_p, [res_mle_local[j].minimizer[1]], subplot=1) 
    StatsPlots.vline!(tmp_p, [res_mle_local[j].minimizer[2]], subplot=2)
    StatsPlots.vline!(tmp_p, [res_mle_local[j].minimizer[3]], subplot=3)
    StatsPlots.vline!(tmp_p, [res_mle_local[j].minimizer[4]], subplot=4)

    StatsPlots.vline!(tmp_p, [res_mle_global[j].minimizer[1]], subplot=1) 
    StatsPlots.vline!(tmp_p, [res_mle_global[j].minimizer[2]], subplot=2)
    StatsPlots.vline!(tmp_p, [res_mle_global[j].minimizer[3]], subplot=3)
    StatsPlots.vline!(tmp_p, [res_mle_global[j].minimizer[4]], subplot=4)

    StatsPlots.vline!(tmp_p, [res_map_local[j].minimizer[1]], subplot=1) 
    StatsPlots.vline!(tmp_p, [res_map_local[j].minimizer[2]], subplot=2)
    StatsPlots.vline!(tmp_p, [res_map_local[j].minimizer[3]], subplot=3)
    StatsPlots.vline!(tmp_p, [res_map_local[j].minimizer[4]], subplot=4)

    StatsPlots.vline!(tmp_p, [res_map_global[j].minimizer[1]], subplot=1) 
    StatsPlots.vline!(tmp_p, [res_map_global[j].minimizer[2]], subplot=2)
    StatsPlots.vline!(tmp_p, [res_map_global[j].minimizer[3]], subplot=3)
    StatsPlots.vline!(tmp_p, [res_map_global[j].minimizer[4]], subplot=4)

    push!(p_dens_mle_arr, tmp_p)
end


# ╔═╡ 0aef9d66-cae1-11ec-172c-c5b833dc9bb4
p_dens_mle =Plots.plot(p_dens_mle_arr...)


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


# ╔═╡ 0aef9d8c-cae1-11ec-0c75-1bf3c9f5df2b
# save(datadir("exp_pro", "labeling_parameters_map_2stp_frac_all_corrected_updated.csv"), df_label_pars_map)


# ╔═╡ 0aef9d8c-cae1-11ec-0848-63e7cfd1eafa
# save(datadir("exp_pro", "labeling_parameters_map_2stp_frac_all_corrected_updated.bson"),  df_label_pars_map => df_label_pars_map)


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
# ╠═0aef9cc2-cae1-11ec-1b30-2b0e199c662a
# ╠═0aef9cce-cae1-11ec-273a-4fb94c2a8337
# ╠═0aef9cce-cae1-11ec-10fe-279087a3e435
# ╠═0aef9cd8-cae1-11ec-274b-ad76a1dcf814
# ╠═0aef9cd8-cae1-11ec-19db-5193e044feec
# ╠═0aef9cd8-cae1-11ec-19ca-27c92d88a69b
# ╠═0aef9ce2-cae1-11ec-2e50-cd375a56e5b5
# ╠═0aef9ce2-cae1-11ec-372a-07838eecfb43
# ╠═0aef9cec-cae1-11ec-3a9c-f18ed66cea9b
# ╠═0aef9cf6-cae1-11ec-20a1-55d39fa0bca2
# ╠═0aef9cf6-cae1-11ec-2c31-85021922ee58
# ╠═0aef9d00-cae1-11ec-39a8-09811c80d8a3
# ╠═a5dd9888-fc23-4480-8c5f-981a3fbc64df
# ╠═0aef9d00-cae1-11ec-11f6-254c146c5757
# ╠═0aef9d0a-cae1-11ec-152b-d72930ff7e95
# ╠═0aef9d14-cae1-11ec-26bc-1dbde25fb496
# ╠═0aef9d14-cae1-11ec-0532-67d89a6717bd
# ╠═0aef9d1e-cae1-11ec-25f6-d978adb8c24f
# ╠═0aef9d28-cae1-11ec-3de9-ab5e5c6c3d6f
# ╠═0aef9d28-cae1-11ec-2110-8db9536a141c
# ╠═0aef9d28-cae1-11ec-32f3-23a11568a93a
# ╠═0aef9d34-cae1-11ec-00b0-5f343f7157a9
# ╠═0aef9d34-cae1-11ec-22d2-552b36bab767
# ╠═0aef9d34-cae1-11ec-1b78-3d3a1cbfc31a
# ╠═0aef9d34-cae1-11ec-1f10-fd1772393856
# ╠═0aef9d3c-cae1-11ec-2598-4d9467f6864e
# ╠═0aef9d3c-cae1-11ec-3b29-492a471db6d2
# ╠═0aef9d46-cae1-11ec-08cb-d3af433c1844
# ╠═5c63f5d4-284d-402b-9f04-88503a9abc81
# ╠═0aef9d46-cae1-11ec-11c4-fb27e8670ea0
# ╠═0aef9d50-cae1-11ec-290d-cd6a65172b4e
# ╠═0aef9d50-cae1-11ec-1f1d-4d568c628508
# ╠═0aef9d50-cae1-11ec-18b5-e79e1d6e93f8
# ╠═0aef9d50-cae1-11ec-1d7e-0bbdbba8b961
# ╠═0aef9d5a-cae1-11ec-110f-ad3f37e7b85d
# ╠═0aef9d5a-cae1-11ec-1546-5963eb87a8fd
# ╠═0aef9d5a-cae1-11ec-2944-331eab7f0ab8
# ╠═0aef9d66-cae1-11ec-172c-c5b833dc9bb4
# ╠═0aef9d6e-cae1-11ec-3673-2568b90fd439
# ╠═0aef9d78-cae1-11ec-28b0-5f8679a5704f
# ╠═0aef9d82-cae1-11ec-2e1b-b3e640d5bc81
# ╠═0aef9d82-cae1-11ec-04f2-0f3fea901855
# ╠═0aef9d8c-cae1-11ec-0c75-1bf3c9f5df2b
# ╠═0aef9d8c-cae1-11ec-0848-63e7cfd1eafa
