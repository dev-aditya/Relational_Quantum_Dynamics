using LinearAlgebra
using QuantumOptics
using Plots
plotlyjs()
using Base.Threads
# Define the Basis

b = SpinBasis(1 // 2)
SIGZ = sigmaz(b)
SIGX = sigmax(b)
I = identityoperator(b)

N = 8 # Number of spins
# Define the Hamiltonian

function sig(oper::Operator, k::Int64, size::Int64)
    list_ = [sparse(I) for i in 1:size]
    list_[k] = oper
    return tensor(list_...)
end
Hs = SIGZ
Hc = sum([sig(SIGZ, i, N - 1) for i in 1:N-1])

for i in 1:N-1
    for j in i+1:N-1
        global Hc += sig(SIGX, i, N - 1) * sig(SIGX, j, N - 1)
    end
end
V = sig(SIGX, 1, N) * sum([sig(SIGX, i, N) for i in 2:N])



H = Hs ⊗ identityoperator(Hc) + I ⊗ Hc + V
HC_EIG_E, HC_EIG_V = eigenstates(dense(Hc))
GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
PSI_EIG_E, PSI_EIG_V = eigenstates(dense(Hs))

function Χ(E::Float64, t::Float64)
    return sum([exp(-im * (HC_EIG_E[i] - E) * t) * HC_EIG_V[i] for i in eachindex(HC_EIG_E)])
end

function φ_λ(t::Float64, E::Float64, Ψ::Ket)
    clc_v = Χ(E, t)
    clock_coeff_conj = [dagger(clc_v) * c_basis for c_basis in HC_EIG_V]
    state = PSI_EIG_V[1] * 0.0
    for i in eachindex(PSI_EIG_E)
        coeff = 0.0
        for j in eachindex(HC_EIG_E)
            coeff += clock_coeff_conj[j] * dagger(PSI_EIG_V[i] ⊗ HC_EIG_V[j]) * Ψ
        end
        state += coeff * PSI_EIG_V[i]
    end
    return state
end


T = Base._linspace(0.0, 2 * π, 1000)

for index in eachindex(GLOB_EIG_E)
    global c1 = Vector{ComplexF64}(undef, length(T))
    global c2 = Vector{ComplexF64}(undef, length(T))

    @threads for i in eachindex(T)
        t = T[i]
        ϕ = φ_λ(t, GLOB_EIG_E[index], GLOB_EIG_V[index])
        if norm(ϕ) < 1e-10
            c1[i] = ϕ.data[1]
            c2[i] = ϕ.data[2]
        else
            ϕ = ϕ / norm(ϕ)
            c1[i] = ϕ.data[1]
            c2[i] = ϕ.data[2]
        end
    end
    Ψ = GLOB_EIG_V[index]
    ρ_red = ptrace(dm(Ψ), [i for i=2:N])
    entan = real(entanglement_entropy(Ψ, [i for i=2:N]))
    plot(T, abs.(c1), label=("Energy: " * string(GLOB_EIG_E[index])), legend=:right)
    plot!(T, abs.(c2), label=("Energy: " * string(entan)), legend=:right, figsize=(12, 8))
    xlabel!("t")
    ylabel!("|c1|^2, |c2|^2")
    title!("Time evolution of the first two states of the system for Index: " * string(index))
    savefig("data/Time evolution of the first two states of the system for Index: " * string(index) * ".png",)
end






