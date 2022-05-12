function U_smooth(t, fr, delta, beta, tau; s=100.0)
    a = tanh(s * (t - tau))
    return exp(-delta * t) * (beta - fr) + fr * ((a + 1) * exp(-delta*(t - tau)) + (1 - a)) / 2
end


function U_step_2stp(t, fr, delta, beta, tau1, tau2, frac)
    return ifelse(t <= tau2, ifelse(t <= tau1, fr*(1-exp(-delta*t)), frac*fr*(1-exp(-delta*(t-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1))),(frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))
end


function U_smooth_2stp(t, fr, delta, beta, tau1, tau2)
    a = tanh(100.0 * (t - tau2))
    b = tanh(100.0 * (t - tau1))
    return (b+1)/2 * (exp(-delta * (t-tau1)) * (beta - fr) + fr * ((a + 1) * exp(-delta*(t - tau2)) + (1 - a)) / 2) 
end

function U_smooth_2stp(t, fr, delta, tau1, tau2, frac)
    f1(t, fr, delta, tau1, tau2, frac) = ((frac*fr*(1-exp(-delta*(tau2-tau1)))+fr*(1-exp(-delta*tau1))*exp(-delta*(tau2-tau1)))*exp(-delta*(t-tau2)))
    
    f2(t, fr, delta, tau1, frac) = (frac*fr*(1-exp(-delta*(t-tau1)))+ fr*(1-exp(-delta*tau1))*exp(-delta*(t-tau1)))
    
    f3(t, fr, delta) = (fr*(1-exp(-delta*t)))

    f_ab(t, tau) = (tanh(1e4 * (t - tau))/2)

    a = f_ab(t, tau2)
    b = f_ab(t, tau1)

     return ((a+0.5)*f1(t, fr, delta, tau1, tau2, frac)+(-a+0.5)*((b+0.5)*f2(t, fr, delta, tau1, frac)+(-b+0.5)*f3(t, fr, delta)))
end