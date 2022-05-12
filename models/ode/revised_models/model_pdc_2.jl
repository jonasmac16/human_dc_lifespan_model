h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(2)

function _model_pdc_2(du,u,h,p,t, U_func, R)
    LpDCm, LpDCb = u
    @unpack R_pDC = R
    p_pDCbm, δ_pDCbm, δ_pDCb, λ_pDC, tau = p[1:(end-4)]
    fr, delta, frac, tau_u = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau_u)

    u1_past = h(p, t-tau; idxs=1)

    du[1] = p_pDCbm * U - (δ_pDCbm + λ_pDC) * LpDCm ##pDCm
    du[2] = λ_pDC * R_pDC * u1_past - δ_pDCb * LpDCb ##pDCb
end