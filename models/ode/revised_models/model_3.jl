function _model_3(du,u,p,t, U_func, R)
    LASDCm, LcDC1m, LDC2m, LASDCb, LcDC1b, LDC2b = u
    @unpack R_cDC1, R_DC2, R_ASDC, R_precDC1bm, R_precDC1b, R_preDC2bm, R_preDC2b = R
    p_ASDCbm, δ_ASDCbm, p_cDC1bm, δ_cDC1bm, p_DC2bm, δ_DC2bm, δ_ASDCb, δ_cDC1b, δ_DC2b, λ_ASDC, λ_cDC1, λ_DC2 = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_ASDCbm * U - (δ_ASDCbm + λ_ASDC) * LASDCm ##ASDCm

    du[2] = p_cDC1bm * U - (δ_cDC1bm + λ_cDC1) * LcDC1m ##cDC1m

    du[3] = p_DC2bm * U - (δ_DC2bm + λ_DC2) * LDC2m ##DC2m

    du[4] = λ_ASDC * R_ASDC * LASDCm - δ_ASDCb * LASDCb ##ASDCb

    du[5] = λ_cDC1 * R_cDC1 * LcDC1m - δ_cDC1b*LcDC1b ##cDC1b

    du[6] = λ_DC2 * R_DC2 * LDC2m - δ_DC2b*LDC2b ##DC2b
end