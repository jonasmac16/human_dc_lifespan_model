function _model_dc3_1(du,u,p,t, U_func, R)
    LDC3m, LDC3b = u
    @unpack R_DC3 = R
    p_DC3bm, δ_DC3bm, δ_DC3b, λ_DC3 = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_DC3bm * U - (δ_DC3bm + λ_DC3) * LDC3m ##DC3m

    du[2] = λ_DC3 * R_DC3 * LDC3m - δ_DC3b * LDC3b ##DC3b
end