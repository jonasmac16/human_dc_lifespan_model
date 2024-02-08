## Implementation by tkf from https://discourse.julialang.org/t/how-to-launch-several-run-cmd-in-parallel/68862/2

function parallel_run(commands; ntasks = Sys.CPU_THREADS)
    request = Channel{Cmd}() do request
        for cmd in commands
            put!(request, cmd)
        end
    end
    @sync for _ in 1:ntasks
        @async try
            foreach(run, request)
        finally
            close(request)  # shutdown on error
        end
    end
end