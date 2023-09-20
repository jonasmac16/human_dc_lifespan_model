function _model_1(du,u,p,t, U_func, R)
    LASDCm, LcDC1m, LcDC2m, LASDCb, LcDC1b, LcDC2b = u
    @unpack R_cDC1, R_cDC2, R_ASDC, R_precDC1bm, R_precDC1b, R_precDC2bm, R_precDC2b = R
    p_ASDCbm, δ_ASDCbm, p_cDC1bm, δ_cDC1bm, p_cDC2bm, δ_cDC2bm, δ_ASDCb, δ_cDC1b, δ_cDC2b, λ_ASDC, λ_cDC1, λ_cDC2, Δ_cDC1bm, Δ_cDC2bm, Δ_cDC1b, Δ_cDC2b = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_ASDCbm * U - (δ_ASDCbm + Δ_cDC1bm + Δ_cDC2bm + λ_ASDC) * LASDCm ##ASDCm

    du[2] = p_cDC1bm * U - (δ_cDC1bm + λ_cDC1) * LcDC1m + Δ_cDC1bm * R_precDC1bm * LASDCm ##cDC1m

    du[3] = p_cDC2bm * U - (δ_cDC2bm + λ_cDC2) * LcDC2m + Δ_cDC2bm * R_precDC2bm * LASDCm ##cDC2m

    du[4] = λ_ASDC * R_ASDC * LASDCm - (δ_ASDCb + Δ_cDC1b + Δ_cDC2b) * LASDCb ##ASDCb

    du[5] = λ_cDC1 * R_cDC1 * LcDC1m + Δ_cDC1b * R_precDC1b * LASDCb - δ_cDC1b*LcDC1b ##cDC1b

    du[6] = λ_cDC2 * R_cDC2 * LcDC2m + Δ_cDC2b * R_precDC2b * LASDCb - δ_cDC2b*LcDC2b ##cDC2b
end