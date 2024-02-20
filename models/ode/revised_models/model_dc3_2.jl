function _model_dc3_2(du,u,p,t, U_func, R)
    LPRO, LDC3m, LDC3b = u
    @unpack R_DC3 = R
    p_PRO, p_DC3bm, δ_PRO, δ_DC3bm, δ_DC3b, ϵ, λ_DC3, R_PRO = p[1:(end-4)]
    fr, delta, frac, tau = p[(end-3):end]

    U = U_func(t, fr, delta, frac, tau)

    du[1] = p_PRO * U - (δ_PRO + ϵ) * LPRO  ##progenitor

    du[2] = p_DC3bm * U + ϵ * LPRO * R_PRO  - (δ_DC3bm + λ_DC3) * LDC3m ##DC3m

    du[3] = λ_DC3 * R_DC3 * LDC3m - δ_DC3b * LDC3b ##DC3b
end