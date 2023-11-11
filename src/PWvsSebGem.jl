include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Plots
using Base.Threads
using LaTeXStrings

include("hamiltonian/highlyCoupled_spinx_spinx.jl")

HC_EIG_E_loc, HC_EIG_V_loc = eigenstates(dense(Hc));
function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
end
quant_system = SpinQuantSystem(Hs, Hc, V, χ);
A = sigmaz(b)
T = Base._linspace(0.0, 2 * π, 1000);
PW_A = Vector{Float64}(undef, length(T))
SebGem_A = Vector{Float64}(undef, length(T))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    @threads for i in eachindex(T)
        t = T[i]
        ϕ = φ_λ(t, quant_system)
        ϕ = ϕ / norm(ϕ)
        global SebGem_A[i] = abs2(expect(A, ϕ))
        proj_ = projector(χ(t, quant_system.EΨ))

        global PW_A[i] = abs2(tr((A ⊗ proj_) * projector(quant_system.Ψ)) / tr((identityoperator(A) ⊗ proj_) * projector(quant_system.Ψ)))
    end
    plot(T, SebGem_A, label=("SebGem"), legend=:right)
    plot!(T, PW_A, label=("PW"), legend=:right, figsize=(19.20, 15.80))
    xlabel!("t")
    ylabel!(L"\langle \sigma_z \rangle")
    savefig("data/HigCop_SxSx_Eigindex_$index" * "sebgemVSpw" * ".png",)
end