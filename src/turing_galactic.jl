### New functions and types ####################################################################################
struct constrained end
struct unconstrained end


function (f::Turing.OptimLogDensity)(G, z)
  spl = DynamicPPL.SampleFromPrior()
  
  # Calculate log joint and the gradient
  l, g = Turing.gradient_logp(
      z, 
      DynamicPPL.VarInfo(f.vi, spl, z), 
      f.model, 
      spl,
      f.context
  )

  # Use the negative gradient because we are minimizing.
  G[:] = -g

  return nothing
end



function transform!(f::Turing.OptimLogDensity)
  spl = DynamicPPL.SampleFromPrior()

  ## Check link status of vi in OptimLogDensity
  linked = DynamicPPL.islinked(f.vi, spl) 

  ## transform into constrained or unconstrained space depending on current state of vi
  if !linked
    f.vi[spl] = f.vi[DynamicPPL.SampleFromPrior()]
    DynamicPPL.link!(f.vi, spl)
  else
    DynamicPPL.invlink!(f.vi, spl)
  end

  return nothing
end


function transform2constrained(par::AbstractArray, vi::DynamicPPL.VarInfo)
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)
  
  !linked && DynamicPPL.link!(vi, spl)
  vi[spl] = par
  DynamicPPL.invlink!(vi,spl)
  tpar = vi[spl]
  
  linked && DynamicPPL.link!(vi,spl)

  return tpar
end

function transform2constrained!(par::AbstractArray, vi::DynamicPPL.VarInfo)
  par .= transform2constrained(par, vi)
  return nothing
end

function transform2unconstrained(par::AbstractArray, vi::DynamicPPL.VarInfo)
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)
  linked && DynamicPPL.invlink!(vi, spl)
  vi[spl] = par
  DynamicPPL.link!(vi, spl)
  tpar = vi[spl]
  !linked && DynamicPPL.invlink!(vi, spl)

  return tpar
end

function transform2unconstrained!(par::AbstractArray, vi::DynamicPPL.VarInfo)
  par .= transform2unconstrained(par, vi)
  return nothing
end

function transform(par::AbstractArray, vi::DynamicPPL.VarInfo) ## transforms into the space of vi
  spl = DynamicPPL.SampleFromPrior()

  linked = DynamicPPL.islinked(vi, spl)

  if linked ## if vi is unconstrained transform from 
    tpar = transform2unconstrained(par, vi)
  else
    tpar = transform2constrained(par, vi)
  end

  return tpar
end

function transform!(par::AbstractArray, vi::DynamicPPL.VarInfo)
  par .= transform(par, vi)
  return nothing
end


function _orderpar(par::NamedTuple, vi::DynamicPPL.VarInfo)
  tmp_idx = indexin(collect(keys(vi.metadata)), collect(keys(par)))
  tmp = collect(par)[tmp_idx]
  return tmp, tmp_idx
end


function transform2unconstrained(par::NamedTuple, vi::DynamicPPL.VarInfo; order::Bool=true, array::Bool=true)
  par_sor, par_idx = _orderpar(par, vi)

  transform2unconstrained!(par_sor, vi)

  if order
    if array
      return par_sor
    else
      return (; zip(keys(vi.metadata), par_sor)...)
    end
  else
    if array
      return par_sor[tmp_idx]
    else
      return (; zip(keys(par),par_sor[par_idx])...)
    end
  end
end

function transform2constrained(par::NamedTuple, vi::DynamicPPL.VarInfo; order::Bool=true, array::Bool=true)
  par_sor, par_idx = _orderpar(par, vi)

  transform2constrained!(par_sor, vi)

  if order
    if array
      return par_sor
    else
      return (; zip(keys(vi.metadata), par_sor)...)
    end
  else
    if array
      return par_sor[tmp_idx]
    else
      return (; zip(keys(par),par_sor[par_idx])...)
    end
  end
end

function transform(par::NamedTuple, vi::DynamicPPL.VarInfo; order::Bool=true, array::Bool=true)
  par_sor, par_idx = _orderpar(par, vi)

  transform!(par_sor, vi)

  if order
    if array
      return par_sor
    else
      return (; zip(keys(vi.metadata), par_sor)...)
    end
  else
    if array
      return par_sor[tmp_idx]
    else
      return (; zip(keys(par),par_sor[par_idx])...)
    end
  end
end


function transform2constrained(res::Optim.MultivariateOptimizationResults, vi::DynamicPPL.VarInfo)
  tres = deepcopy(res)
  if !any(isnan.(tres.minimizer))
    tres.minimizer = transform2constrained(tres.minimizer, vi)
  else
      @warn "Could not transform optimisation results due to NaNs in ':minimizer'."
  end

  if !any(isnan.(tres.initial_x))
    tres.initial_x = transform2constrained(tres.initial_x, vi)
  else
    @warn "Could not transform initial values due to NaNs in ':initial_x'."
  end

  return tres
end

function transform2constrained!(res::Optim.MultivariateOptimizationResults, vi::DynamicPPL.VarInfo)

  if !any(isnan.(res.minimizer))
    res.minimizer = transform2constrained(res.minimizer, vi)
  else
      @warn "Could not transform optimisation results due to NaNs in ':minimizer'."
  end

  if !any(isnan.(res.initial_x))
    res.initial_x = transform2constrained(res.initial_x, vi)
  else
    @warn "Could not transform initial values due to NaNs in ':initial_x'."
  end

  return nothing
end



function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MAP , ::unconstrained; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj = Turing.OptimLogDensity(model, Turing.OptimizationContext(DynamicPPL.DefaultContext()))

  transform!(obj)

  if init_vals === nothing
    init_vals = obj.vi[DynamicPPL.SampleFromPrior()]
  else
    init_vals = transform2unconstrained(init_vals, obj.vi)
  end

  t(res) = transform2constrained(res, obj.vi)

  return (obj=obj, init_vals = init_vals, transform=t)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MAP , ::constrained, lb, ub; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj = Turing.OptimLogDensity(model, Turing.OptimizationContext(DynamicPPL.DefaultContext()))

  if init_vals === nothing
    init_vals = rand.(Uniform.(lb, ub)) # I think we should sample from a uniform within the provided parameter bounds as they might be different to the priors which would come from obj.vi[DynamicPPL.SampleFromPrior()]
  elseif isa(init_vals, NamedTuple)
    init_vals = _orderpar(init_vals, obj.vi)[1]
  end

  return (obj=obj, init_vals = init_vals, transform=identity)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MLE , ::unconstrained; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj = Turing.OptimLogDensity(model, Turing.OptimizationContext(DynamicPPL.LikelihoodContext()))

  transform!(obj)

  if init_vals === nothing
    init_vals = obj.vi[DynamicPPL.SampleFromPrior()]
  else
    init_vals = transform2unconstrained(init_vals, obj.vi)
  end

  t(res) = transform2constrained(res, obj.vi)

  return (obj=obj, init_vals = init_vals, transform=t)
end

function instantiate_optimisation_problem(model::DynamicPPL.Model, ::MLE, ::constrained, lb, ub; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj = Turing.OptimLogDensity(model, Turing.OptimizationContext(DynamicPPL.LikelihoodContext()))

  if init_vals === nothing
    init_vals = rand.(Uniform.(lb, ub)) # I think we should sample from a uniform within the provided parameter bounds as they might be different to the priors which would come from obj.vi[DynamicPPL.SampleFromPrior()]
  elseif isa(init_vals, NamedTuple)
    init_vals = _orderpar(init_vals, obj.vi)[1]
  end

  return (obj=obj, init_vals = init_vals, transform=identity)
end





function instantiate_galacticoptim_problem(model::DynamicPPL.Model, ::MAP , ::unconstrained; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj, init, t =instantiate_optimisation_problem(model, MAP() , unconstrained(); init_vals = init_vals, p = p)
  
  l(x,p) = obj(x)

  f = Optimization.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  prob = Optimization.OptimizationProblem(f, init, p)

  return (prob=prob, transform = t)
end

function instantiate_galacticoptim_problem(model::DynamicPPL.Model, ::MLE , ::unconstrained; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj, init, t =instantiate_optimisation_problem(model, MLE() , unconstrained(); init_vals = init_vals, p = p)
  
  l(x,p) = obj(x)

  f = Optimization.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  prob = Optimization.OptimizationProblem(f, init, p)

  return (prob=prob, transform = t)
end

function instantiate_galacticoptim_problem(model::DynamicPPL.Model, ::MAP , ::constrained, lb, ub; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj, init, t =instantiate_optimisation_problem(model, MAP(), constrained(), lb, ub; init_vals = init_vals, p = p)
  
  l(x,p) = obj(x)

  f = Optimization.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  prob = Optimization.OptimizationProblem(f, init, p;lb=lb, ub=ub)

  return (prob=prob, transform = t)
end

function instantiate_galacticoptim_problem(model::DynamicPPL.Model, ::MLE , ::constrained, lb, ub; init_vals = nothing, p = DiffEqBase.NullParameters())
  obj, init, t =instantiate_optimisation_problem(model, MLE() , constrained(), lb, ub; init_vals = init_vals, p = p)
  
  l(x,p) = obj(x)

  f = Optimization.OptimizationFunction(l, grad = (G,x,p) -> obj(G,x))
  prob = Optimization.OptimizationProblem(f, init, p;lb=lb, ub=ub)

  return (prob=prob, transform = t)
end
#######################################
