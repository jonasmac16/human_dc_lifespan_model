function U_smooth(t, fr, delta, beta, tau; s=100.0, c=0.75)
    a = tanh(s * (t - tau))
    return c * ( exp(-delta * t) * (beta - fr) + fr * ((a + 1) * exp(-delta*(t - tau)) + (1 - a)) / 2 )
end


function U_step_2stp(t, fr, delta, beta, tau1, tau2, frac; c=0.75)
    return c * ( ifelse(t <= tau2, ifelse(t <= tau1, fr*(1-exp(-delta*t)), frac*fr*(1-exp(-delta*(t-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1))),(frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2))) )
end



function U_smooth_2stp(t, fr, delta, tau1, tau2, frac; c=0.75)
    f1(t, fr, delta, tau1, tau2, frac) = ((frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))
    
    f2(t, fr, delta, tau1, frac) = (frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))
    
    f3(t, fr, delta) = (fr*(1-exp(-delta*t)))

    f_ab(t, tau) = (tanh(1e4 * (t - tau))/2)

    a = f_ab(t, tau2)
    b = f_ab(t, tau1)

     return c* ((a+0.5)*f1(t, fr, delta, tau1, tau2, frac)+(-a+0.5)*((b+0.5)*f2(t, fr, delta, tau1, frac)+(-b+0.5)*f3(t, fr, delta)))
end


# function U_smooth_2stp(t, fr, delta, beta, tau1, tau2; c=0.75)
#     a = tanh(100.0 * (t - tau2))
#     b = tanh(100.0 * (t - tau1))
#     return c * ( (b+1)/2 * (exp(-delta * (t-tau1)) * (beta - fr) + fr * ((a + 1) * exp(-delta*(t - tau2)) + (1 - a)) / 2)  )
# end


# function U_step_2stp_split(t, fr, delta, beta, tau1, tau2, frac; c=0.75)
    
#     if t <= tau2
#         if t <= tau1
#             fr*(1-exp(-delta*t))
#         else
#             frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))
#         end
#     else
#         # frac*fr*(1-exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2))
#         # +
#         # fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))


#         (frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))      *exp(-delta*(t-tau2))
#     end


    
#     ( ifelse(t <= tau2,
#      ifelse(t <= tau1, fr*(1-exp(-delta*t)), frac*fr*(1-exp(-delta*(t-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1))),
#        (frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2))) )
    
#     return c * 
# end

# using DataFrames


# function test(t, fr, delta, beta, tau1, tau2, frac; c=0.75)
#     (frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))      *exp(-delta*(t-tau2))
# end


# function test2(t, fr, delta, beta, tau1, tau2, frac; c=0.75)
#     frac*fr*(1-exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)) + fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1))
# end



# label_ps = DataFrame(load(datadir("exp_pro", "labeling_parameters_frac.csv")))
# fr = label_ps[1,:fr]
# delta = label_ps[1,:delta]
# frac = label_ps[1,:frac]
# beta = 0.0
# tau1 = 0.5/24.0
# tau2 = 3.0/24.0
# tp = tau1+tau1
# tps= collect(0.0:0.01:2.0)

# function U_smooth_2stp_new(t, fr, delta, beta, tau1, tau2, frac; c=0.75)
#     a = tanh(1e4 * (t - tau2))/2
#     b = tanh(1e4 * (t - tau1))/2

#      return c* ((a+0.5)*((frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))+(-a+0.5)*((b+0.5)*(frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))+(-b+0.5)*(fr*(1-exp(-delta*t)))))
#     # return ((-a+0.5)*(1.0)+(a+0.5)*2.0, (a+0.5),(b+0.5), (-b+0.5), (-a+0.5))
# end

# function U_smooth_2stp_func(t, fr, delta, tau1, tau2, frac; c=0.75)
#     a = tanh(1e4 * (t - tau2))/2
#     b = tanh(1e4 * (t - tau1))/2
#     function f1(t, fr, delta, tau1, tau2, frac)
#         return ((frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))
#     end

#     function f2(t, fr, delta, tau1, frac)
#         return (frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))
#     end

#     function f3(t, fr, delta)
#         return (fr*(1-exp(-delta*t)))
#     end

#      return c* ((a+0.5)*f1(t, fr, delta, tau1, tau2, frac)+(-a+0.5)*((b+0.5)*f2(t, fr, delta, tau1, frac)+(-b+0.5)*f3(t, fr, delta)))
# end

# function f1g(t, fr, delta, tau1, tau2, frac)
#     return ((frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))
# end

# function f2g(t, fr, delta, tau1, frac)
#     return (frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))
# end

# function f3g(t, fr, delta)
#     return (fr*(1-exp(-delta*t)))
# end


# function U_smooth_2stp_anon_func(t, fr, delta, tau1, tau2, frac; c=0.75)
#     a = tanh(1e4 * (t - tau2))/2
#     b = tanh(1e4 * (t - tau1))/2
    
#     f1(t, fr, delta, tau1, tau2, frac) = ((frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))
    
#     f2(t, fr, delta, tau1, frac) = (frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))
    
#     f3(t, fr, delta) = (fr*(1-exp(-delta*t)))
    
#      return c* ((a+0.5)*f1(t, fr, delta, tau1, tau2, frac)+(-a+0.5)*((b+0.5)*f2(t, fr, delta, tau1, frac)+(-b+0.5)*f3(t, fr, delta)))
# end

# function U_smooth_2stp_gfunc(t, fr, delta, tau1, tau2, frac; c=0.75)
#     a = tanh(1e4 * (t - tau2))/2
#     b = tanh(1e4 * (t - tau1))/2

#      return c* ((a+0.5)*f1g(t, fr, delta, tau1, tau2, frac)+(-a+0.5)*((b+0.5)*f2g(t, fr, delta, tau1, frac)+(-b+0.5)*f3g(t, fr, delta)))
# end

# function f_ab(t, tau)
#     return tanh(1e4 * (t - tau))/2
# end



# function U_smooth_2stp_gfunc_all(t, fr, delta, tau1, tau2, frac; c=0.75)
#     a = f_ab(t, tau2)
#     b = f_ab(t, tau1)

#      return c* ((a+0.5)*f1g(t, fr, delta, tau1, tau2, frac)+(-a+0.5)*((b+0.5)*f2g(t, fr, delta, tau1, frac)+(-b+0.5)*f3g(t, fr, delta)))
# end


# using Plots
# @benchmark U_smooth_2stp_new(tp, fr, delta, beta, tau1, tau2, frac)
# @benchmark U_step_2stp(tp, fr, delta, beta, tau1, tau2, frac)
# @benchmark U_smooth_2stp_func(tp, fr, delta, tau1, tau2, frac)
# @benchmark U_smooth_2stp_anon_func(tp, fr, delta, tau1, tau2, frac)
# @benchmark U_smooth_2stp_gfunc(tp, fr, delta, tau1, tau2, frac)
# @benchmark U_smooth_2stp_gfunc_all(tp, fr, delta, tau1, tau2, frac)




# plot(tps, [U_step_2stp(j, fr, delta, beta, tau1, tau2, frac) for j in tps])
# plot!(tps, [U_smooth_2stp_new(j, fr, delta, beta, tau1, tau2, frac) for j in tps])
# plot!(tps, [U_smooth_2stp_func(j, fr, delta, tau1, tau2, frac) for j in tps])
# plot!(tps, [U_smooth_2stp_anon_func(j, fr, delta, tau1, tau2, frac) for j in tps])


# U_smooth_2stp_new(0.0, fr, delta, beta, tau1, tau2, frac)


# U_step_2stp(0.0, fr, delta, beta, tau1, tau2, frac)[1]

# test(tp, fr, delta, beta, tau1, tau2, frac) == test2(tp, fr, delta, beta, tau1, tau2, frac)

# test(tp, fr, delta, beta, tau1, tau2, frac) ≈ test2(tp, fr, delta, beta, tau1, tau2, frac)



