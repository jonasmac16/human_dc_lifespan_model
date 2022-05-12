function df2latex(df, filename; booktabs=true, env=:table, kwargs...)
    latex_string = latexify(df, booktabs=booktabs, env=env,kwargs...)
    write(filename,latex_string)
end
