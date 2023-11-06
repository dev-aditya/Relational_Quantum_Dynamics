using QuantumOptics
using Base.Threads
using Plots

abstract type QuantumSystem end

mutable struct SpinQuantSystem <: QuantumSystem
    Hs::Operator
    Hc::Operator
    V::Operator
    clock_func::Function
    index::Union{Int, Nothing}
    HC_EIG_E
    HC_EIG_V
    GLOB_EIG_E
    GLOB_EIG_V
    PSI_EIG_E
    PSI_EIG_V
    Ψ::Ket
    EΨ::Float64
    function SpinQuantSystem(Hs::Operator, Hc::Operator, V::Operator, clock_func::Function)
        system = new()
        system.Hs = Hs
        system.Hc = Hc
        system.V = V
        system.clock_func = clock_func
        system.index = nothing
        H = system.Hs ⊗ identityoperator(system.Hc) + identityoperator(system.Hs) ⊗ system.Hc + system.V
        system.HC_EIG_E, system.HC_EIG_V = eigenstates(dense(system.Hc))
        system.GLOB_EIG_E, system.GLOB_EIG_V = eigenstates(dense(H))
        system.PSI_EIG_E, system.PSI_EIG_V = eigenstates(dense(system.Hs))
        if isnothing(system.index)
            system.index = 1
        end
        system.Ψ = system.GLOB_EIG_V[system.index]
        system.EΨ = system.GLOB_EIG_E[system.index]
        return system
    end
end

function calculate_eigenstates(system::SpinQuantSystem)
    H = system.Hs ⊗ identityoperator(system.Hc) + identityoperator(system.Hs) ⊗ system.Hc + system.V
    HC_EIG_E, HC_EIG_V = eigenstates(dense(system.Hc))
    GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
    PSI_EIG_E, PSI_EIG_V = eigenstates(dense(system.Hs))
    return HC_EIG_E, HC_EIG_V, GLOB_EIG_E, GLOB_EIG_V, PSI_EIG_E, PSI_EIG_V
end

function UpdateIndex(System::SpinQuantSystem, index::Int64)
    System.index = index
    System.Ψ = System.GLOB_EIG_V[index]
    System.EΨ = System.GLOB_EIG_E[index]
    println("Index updated to $index")
end


function φ_λ(t::Float64, system::SpinQuantSystem)
    clc_v = system.clock_func(t, system.EΨ)
    clock_coeff_conj = [dagger(clc_v)*c_basis for c_basis in system.HC_EIG_V]
    state = system.PSI_EIG_V[1]*0.0
    for i in eachindex(system.PSI_EIG_E)
        coeff = 0.0
        for j in eachindex(system.HC_EIG_E)
            coeff = coeff + clock_coeff_conj[j] * dagger(system.PSI_EIG_V[i] ⊗ system.HC_EIG_V[j]) * system.Ψ
        end
        state += coeff * system.PSI_EIG_V[i]
    end
    return state
end

# %%
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

HC_EIG_E_loc, HC_EIG_V_loc = eigenstates(dense(Hc))
function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
end

quant_system = SpinQuantSystem(Hs, Hc, V, χ)

T = Base._linspace(0.0, 2 * π, 1000)
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    global c1 = Vector{ComplexF64}(undef, length(T))
    global c2 = Vector{ComplexF64}(undef, length(T))

   @threads for i in eachindex(T)
        t = T[i]
        ϕ = φ_λ(t, quant_system)
        if norm(ϕ) < 1e-10
            c1[i] = ϕ.data[1]
            c2[i] = ϕ.data[2]
        else
            ϕ = ϕ / norm(ϕ)
            c1[i] = ϕ.data[1]
            c2[i] = ϕ.data[2]
        end
    end
    entan = real(entanglement_entropy(quant_system.Ψ, [i for i=2:N]))/log(2)
    plot(T, abs.(c1), label=("Energy: " * string(quant_system.EΨ)), legend=:right)
    plot!(T, abs.(c2), label=("Entangement: " * string(entan)), legend=:right, figsize=(19.20, 15.80))
    xlabel!("t")
    ylabel!("|c1|^2, |c2|^2")
    title!("Time evolution of the states of the system for Index: " * string(index))
    savefig("data/Time evolution of thestates of the system for Index: " * string(index) * ".png",)
end