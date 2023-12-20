function _model_1(du,u,p,t, U_func, R)
    LASDCm, LcDC1m, LDC2m, LASDCb, LcDC1b, LDC2b = u
    @unpack RcDC1, RDC2, RASDC, RASDC_cDC1_bm, RASDC_cDC1_blood, RASDC_DC2_bm, RASDC_DC2_blood = R
    p_ASDCbm, δ_ASDCbm, p_cDC1bm, δ_cDC1bm, p_DC2bm, δ_DC2bm, δ_ASDCb, δ_cDC1b, δ_DC2b, λ_ASDC, λ_cDC1, λ_DC2, Δ_cDC1bm, Δ_DC2bm, Δ_cDC1b, Δ_DC2b = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_ASDCbm * U - (δ_ASDCbm + Δ_cDC1bm + Δ_DC2bm + λ_ASDC) * LASDCm ##ASDCm

    du[2] = p_cDC1bm * U - (δ_cDC1bm + λ_cDC1) * LcDC1m + Δ_cDC1bm * RASDC_cDC1_bm * LASDCm ##cDC1m

    du[3] = p_DC2bm * U - (δ_DC2bm + λ_DC2) * LDC2m + Δ_DC2bm * RASDC_DC2_bm * LASDCm ##DC2m

    du[4] = λ_ASDC * RASDC * LASDCm - (δ_ASDCb + Δ_cDC1b + Δ_DC2b) * LASDCb ##ASDCb

    du[5] = λ_cDC1 * RcDC1 * LcDC1m + Δ_cDC1b * RASDC_cDC1_blood * LASDCb - δ_cDC1b*LcDC1b ##cDC1b

    du[6] = λ_DC2 * RDC2 * LDC2m + Δ_DC2b * RASDC_DC2_blood * LASDCb - δ_DC2b*LDC2b ##DC2b
end