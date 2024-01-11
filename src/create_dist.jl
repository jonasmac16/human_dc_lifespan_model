function create_dist(dist::AbstractString, mu::Real, sigma::Real, truncated::Real, lower=-Inf, upper=Inf)
	mu = string(mu)
	sigma = string(sigma)
	lower = string(lower)
	upper = string(upper)
	
	dist_string = dist*"("*mu*", "*sigma*")"
	if truncated == 1
		dist_string = "truncated("*dist_string*","*lower*","*upper*")"
	end
	
	return eval(Meta.parse(dist_string))
end
