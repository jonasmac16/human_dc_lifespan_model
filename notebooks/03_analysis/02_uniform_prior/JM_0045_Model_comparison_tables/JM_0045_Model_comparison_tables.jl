using DrWatson
DrWatson.@quickactivate "Model of DC Differentiation"
using CSV
using CairoMakie
using AlgebraOfGraphics
using DataFrames
using Pipe
using CategoricalArrays


dc3_loo_df = CSV.read(projectdir("notebooks","03_analysis","02_uniform_prior", "JM_0043_Julia_Analysis_DC3", "results", "PSIS_LOO_CV_Model_comparison_DC3_leave_out_sample.csv"), DataFrame)
asdc_loo_df = CSV.read(projectdir("notebooks","03_analysis","02_uniform_prior", "JM_0042_Julia_Analysis_ASDC_cDC1_DC2", "results", "PSIS_LOO_CV_Model_comparison_leave_out_sample_extended.csv"), DataFrame)
asdc_loo_subsets_df = CSV.read(projectdir("notebooks","03_analysis","02_uniform_prior", "JM_0042_Julia_Analysis_ASDC_cDC1_DC2", "results", "PSIS_LOO_CV_Model_comparison_leave_out_subset_extended.csv"), DataFrame)


function make_loo_df(df)
    @pipe df |>
    sort(_, :loo, rev=true) |>
    transform(_, AsTable([:loo, :d_loo]) => (x -> first(x.loo) .- x.d_loo) => :scaled_d_loo) |>
    select(_, [:name, :rank, :loo, :se, :scaled_d_loo, :dse]) |>
    rename(_, :scaled_d_loo => :d_loo) |>
    transform(_, :name => (x -> categorical(x, levels=x, compress=true)), renamecols=false)
end

dc3_loo_df_scaled = make_loo_df(dc3_loo_df)
asdc_loo_df_scaled = make_loo_df(asdc_loo_df)
asdc_loo_subsets_df_scaled = make_loo_df(asdc_loo_subsets_df)

dc3_loo_df_striped = @pipe dc3_loo_df_scaled |> select(_, Not([:d_loo, :dse]))
asdc_loo_df_striped = @pipe asdc_loo_df_scaled |> select(_, Not([:d_loo, :dse]))
set_aog_theme!()

function df_plot(df)
    f = CairoMakie.Figure(; resolution=(600,400))
    ax = Axis(f[1,1])

    CairoMakie.errorbars!(ax,df.loo,  collect(1:nrow(df)), df.se, direction=:x, color=:black,linewidth=2, whiskerwidth=8)

    @pipe df |>
    data(_) * mapping(:loo, :name, color=:name, group=:name) * visual(Scatter, markersize=10) |>
    draw!(ax, _; palettes=(color=cgrad(:roma,nrow(df), categorical=true),))

    f

    CairoMakie.errorbars!(ax,df.loo[2:end],  collect(2:nrow(df)).+ 0.1, df.dse[2:end], direction=:x, whiskerwidth=8, linewidth=2, color=:grey)
    CairoMakie.scatter!(ax, df.d_loo[2:end], collect(2:nrow(df)) .+ 0.1, color=:grey, markersize=8)

    CairoMakie.vlines!(ax, maximum(df.loo), color=:grey, linestyle = :dash, linewidth=1)
    ax.ylabel =""
    ax.xlabel ="elpd (greater is better)"

    return f
end

df_plot(dc3_loo_df_scaled)
df_plot(asdc_loo_df_scaled)
df_plot(asdc_loo_subsets_df_scaled)

mkpath(projectdir("notebooks", "03_analysis","02_uniform_prior", basename(@__DIR__), "results"))

CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior", basename(@__DIR__), "results","dc3_loo_sample.csv"), dc3_loo_df_scaled)
CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior", basename(@__DIR__), "results","asdc_loo_sample.csv"), asdc_loo_df_scaled)
CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior", basename(@__DIR__), "results","asdc_loo_subsets.csv"), asdc_loo_subsets_df_scaled)

CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior", basename(@__DIR__), "results","dc3_loo_sample_paper.csv"), dc3_loo_df_striped)
CSV.write(projectdir("notebooks", "03_analysis","02_uniform_prior", basename(@__DIR__), "results","asdc_loo_sample_paper.csv"), asdc_loo_df_striped)