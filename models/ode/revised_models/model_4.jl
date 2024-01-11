function _model_4(du,u,p,t, U_func, R)
    LpreDCm, LcDC1m, LcDC2m, LpreDCb, LcDC1b, LcDC2b = u
    @unpack R_cDC1, R_cDC2, R_preDC, R_precDC1bm, R_precDC1b, R_precDC2bm, R_precDC2b = R
    p_ASDCbm, δ_ASDCbm, p_cDC1bm, δ_cDC1bm, p_DC2bm, δ_DC2bm, δ_ASDCb, δ_cDC1b, δ_DC2b, λ_ASDC, Δ_cDC1bm, Δ_DC2bm, Δ_cDC1b, Δ_DC2b = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_ASDCbm * U - (δ_ASDCbm + Δ_cDC1bm + Δ_DC2bm) * LpreDCm ##preDCm

    du[2] = p_cDC1bm * U - δ_cDC1bm * LcDC1m + Δ_cDC1bm * R_precDC1bm * LpreDCm ##cDC1m

    du[3] = p_DC2bm * U - δ_DC2bm  * LcDC2m + Δ_DC2bm * R_precDC2bm * LpreDCm ##cDC2m

    du[4] = λ_ASDC * R_preDC * LpreDCm - (δ_ASDCb + Δ_cDC1b + Δ_DC2b) * LpreDCb ##preDCb

    du[5] = Δ_cDC1b * R_precDC1b * LpreDCb - δ_cDC1b*LcDC1b ##cDC1b

    du[6] = Δ_DC2b * R_precDC2b * LpreDCb - δ_DC2b*LcDC2b ##cDC2b
end