function _model_pdc_1(du,u,p,t, U_func, R)
    LpDCm, LpDCb = u
    @unpack R_pDC = R
    p_DC3bm, δ_DC3bm, δ_DC3b, λ_DC3 = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_DC3bm * U - (δ_DC3bm + λ_DC3) * LpDCm ##pDCm

    du[2] = λ_DC3 * R_pDC * LpDCm - δ_DC3b * LpDCb ##pDCb
end