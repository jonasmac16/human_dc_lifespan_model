### A Pluto.jl notebook ###
# v0.16.4

using Markdown
using InteractiveUtils

# ╔═╡ bc77b832-d71e-11ee-23c6-6da27f9f71ff
using DrWatson

# ╔═╡ 4590a081-03eb-474b-8b17-d549610fcac3
DrWatson.@quickactivate "Model of DC Differentiation"

# ╔═╡ 3317e814-37e1-489a-ae3b-1c3e6b516d7c
begin
	using CSV
	using CairoMakie
	using AlgebraOfGraphics
	using DataFrames
	using Pipe
	using CategoricalArrays
end

# ╔═╡ 219c96aa-a932-4141-9b27-3273af096d30
mkpath(projectdir("notebooks", "03_analysis","02_uniform_prior/", basename(@__DIR__), "results"))

# ╔═╡ 912b3165-551e-4569-b8d7-8e52dfc098dd
AlgebraOfGraphics.set_aog_theme!()

# ╔═╡ 87e318de-e20e-4f4c-ae5c-60a3684860eb
function make_loo_df(df)
    @pipe df |>
    sort(_, :elpd_loo, rev=true) |>
    transform(_, AsTable([:elpd_loo, :elpd_diff]) => (x -> first(x.elpd_loo) .- x.elpd_diff) => :scaled_d_loo) |>
    select(_, [:name, :rank, :elpd_loo, :se, :scaled_d_loo, :dse]) |>
    rename(_, :scaled_d_loo => :d_loo) |>
    transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)
end

# ╔═╡ d6b7806c-3499-4cb0-8357-8fc4d5cb54e4
function df_plot(df)
    f = CairoMakie.Figure(; resolution=(600,400))
    ax = Axis(f[1,1])

    CairoMakie.errorbars!(ax,df.elpd_loo,  collect(1:nrow(df)), df.se, direction=:x, color=:black,linewidth=2, whiskerwidth=8)

    @pipe df |>
    data(_) * mapping(:elpd_loo, :name, color=:name, group=:name) * visual(Scatter, markersize=10) |>
    draw!(ax, _; palettes=(color=cgrad(:roma,nrow(df), categorical=true),))

    f

    CairoMakie.errorbars!(ax,df.elpd_loo[2:end],  collect(2:nrow(df)).+ 0.1, df.dse[2:end], direction=:x, whiskerwidth=8, linewidth=2, color=:grey)
    CairoMakie.scatter!(ax, df.elpd_loo[2:end], collect(2:nrow(df)) .+ 0.1, color=:grey, markersize=8)

    CairoMakie.vlines!(ax, maximum(df.elpd_loo), color=:grey, linestyle = :dash, linewidth=1)
    ax.ylabel =""
    ax.xlabel ="elpd (greater is better)"

    return f
end


# ╔═╡ e61cfcba-a6b6-4e5b-93b1-95d1fcdac5cf
dc3_loo_df = CSV.read(projectdir("notebooks","03_analysis","02_uniform_prior/", "JM_0043_Julia_Analysis_DC3", "results", "PSIS_LOO_CV_Model_comparison_DC3_leave_out_sample.csv"), DataFrame)

# ╔═╡ f9ec175f-1853-4a34-9fd4-7ae09cb39527
asdc_loo_df = CSV.read(projectdir("notebooks","03_analysis","02_uniform_prior/", "JM_0042_Julia_Analysis_ASDC_cDC1_DC2", "results", "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.csv"), DataFrame)

# ╔═╡ c33d9a2f-2574-49b0-9e5c-d2ad059d0e45
asdc_loo_subsets_df = CSV.read(projectdir("notebooks","03_analysis","02_uniform_prior/", "JM_0042_Julia_Analysis_ASDC_cDC1_DC2", "results", "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.csv"), DataFrame)

# ╔═╡ 2e75728c-db4b-48c9-8249-72164a53e168
dc3_loo_df_scaled = make_loo_df(dc3_loo_df)

# ╔═╡ 9f3ccb2d-998b-4750-86a8-bf7c0ac91b57
asdc_loo_df_scaled = make_loo_df(asdc_loo_df)

# ╔═╡ 8f76c54c-e799-40e6-87fc-17b77b937dc7
asdc_loo_subsets_df_scaled = make_loo_df(asdc_loo_subsets_df)

# ╔═╡ f9f1702b-ee91-4ea6-8259-329697cfce24
dc3_loo_df_striped = @pipe dc3_loo_df_scaled |> select(_, Not([:d_loo, :dse]))

# ╔═╡ 340bc71f-0433-4c94-bf07-f84494d9fc39
asdc_loo_df_striped = @pipe asdc_loo_df_scaled |> select(_, Not([:d_loo, :dse]))

# ╔═╡ f8e9c645-8f8f-4980-a248-a230a05f5d72
df_plot(dc3_loo_df_scaled)

# ╔═╡ 80b8fcea-a277-4dba-95a0-d9932c24494c
df_plot(asdc_loo_df_scaled)

# ╔═╡ 6a02da7f-ce0f-43c0-b53e-ca30eb3e26c7
df_plot(asdc_loo_subsets_df_scaled)

# ╔═╡ 72341233-4339-4b42-b546-0836d1b201ee
begin
	CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior/", basename(@__DIR__), "results","dc3_loo_sample.csv"), dc3_loo_df_scaled)
	CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior/", basename(@__DIR__), "results","asdc_loo_sample.csv"), asdc_loo_df_scaled)
	CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior/", basename(@__DIR__), "results","asdc_loo_subsets.csv"), asdc_loo_subsets_df_scaled)

	CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior/", basename(@__DIR__), "results","dc3_loo_sample_paper.csv"), dc3_loo_df_striped)
	CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior/", basename(@__DIR__), "results","asdc_loo_sample_paper.csv"), asdc_loo_df_striped)
end

# ╔═╡ Cell order:
# ╠═bc77b832-d71e-11ee-23c6-6da27f9f71ff
# ╠═4590a081-03eb-474b-8b17-d549610fcac3
# ╠═3317e814-37e1-489a-ae3b-1c3e6b516d7c
# ╠═219c96aa-a932-4141-9b27-3273af096d30
# ╠═912b3165-551e-4569-b8d7-8e52dfc098dd
# ╠═87e318de-e20e-4f4c-ae5c-60a3684860eb
# ╠═d6b7806c-3499-4cb0-8357-8fc4d5cb54e4
# ╠═e61cfcba-a6b6-4e5b-93b1-95d1fcdac5cf
# ╠═f9ec175f-1853-4a34-9fd4-7ae09cb39527
# ╠═c33d9a2f-2574-49b0-9e5c-d2ad059d0e45
# ╠═2e75728c-db4b-48c9-8249-72164a53e168
# ╠═9f3ccb2d-998b-4750-86a8-bf7c0ac91b57
# ╠═8f76c54c-e799-40e6-87fc-17b77b937dc7
# ╠═f9f1702b-ee91-4ea6-8259-329697cfce24
# ╠═340bc71f-0433-4c94-bf07-f84494d9fc39
# ╠═f8e9c645-8f8f-4980-a248-a230a05f5d72
# ╠═80b8fcea-a277-4dba-95a0-d9932c24494c
# ╠═6a02da7f-ce0f-43c0-b53e-ca30eb3e26c7
# ╠═72341233-4339-4b42-b546-0836d1b201ee
