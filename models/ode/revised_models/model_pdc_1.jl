function _model_pdc_1(du,u,p,t, U_func, R)
    LpDCm, LpDCb = u
    @unpack R_pDC = R
    p_pDCbm, δ_pDCbm, δ_pDCb, λ_pDC = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_pDCbm * U - (δ_pDCbm + λ_pDC) * LpDCm ##pDCm

    du[2] = λ_pDC * R_pDC * LpDCm - δ_pDCb * LpDCb ##pDCb
end