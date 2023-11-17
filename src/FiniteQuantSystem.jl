module FiniteQuantSystem
export SpinQuantSystem, UpdateIndex, φ_λ, calculate_eigenstates
using QuantumOptics
abstract type QuantumSystem end
mutable struct SpinQuantSystem <: QuantumSystem
    Hs::Operator
    Hc::Operator
    H::Operator
    V::Operator
    index::Int64
    HC_EIG_E::Vector{Float64}
    HC_EIG_V::Vector{Ket}
    GLOB_EIG_E::Vector{Float64}
    GLOB_EIG_V::Vector{Ket}
    PSI_EIG_E::Vector{Float64}
    PSI_EIG_V::Vector{Ket}
    Ψ::Ket
    EΨ::Float64
    function SpinQuantSystem(Hs::Operator, Hc::Operator, V::Operator)
        println("Initializing SpinQuantSystem")
        system = new()
        system.Hs = Hs
        system.Hc = Hc
        system.V = V
        system.index = 1
        H = system.Hs ⊗ identityoperator(system.Hc) + identityoperator(system.Hs) ⊗ system.Hc + system.V
        system.H = H
        println("Calculating Global Eigenstates \n")
        system.GLOB_EIG_E, system.GLOB_EIG_V = eigenstates(dense(H))
        println("Calculating Clock Eigenstates \n")
        system.HC_EIG_E, system.HC_EIG_V = eigenstates(dense(system.Hc))
        println("Calculating System Eigenstates \n")
        system.PSI_EIG_E, system.PSI_EIG_V = eigenstates(dense(system.Hs))
        println("Setting Default State as the Ground State \n")
        system.Ψ = system.GLOB_EIG_V[system.index]
        system.EΨ = system.GLOB_EIG_E[system.index]
        return system
    end
end

function UpdateIndex(System::SpinQuantSystem, index::Int64)
    System.index = index
    System.Ψ = System.GLOB_EIG_V[index]
    System.EΨ = System.GLOB_EIG_E[index]
    println("EigenIndex updated to $index")
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#= 
The function below has been replaced by a more efficient version:
    Which is simply:
        tensor(I, dagger(χ(t, system.EΨ)))* system.Ψ
    This old version is kept for reference.
=#
function φ_λ(t::Float64, system::SpinQuantSystem)
    clc_v = system.clock_func(t, system.EΨ)
    clock_coeff_conj = [dagger(clc_v) * c_basis for c_basis in system.HC_EIG_V]
    state = system.PSI_EIG_V[1] * 0.0
    for i in eachindex(system.PSI_EIG_E)
        coeff = 0.0
        for j in eachindex(system.HC_EIG_E)
            coeff = coeff + clock_coeff_conj[j] * dagger(system.PSI_EIG_V[i] ⊗ system.HC_EIG_V[j]) * system.Ψ
        end
        state += coeff * system.PSI_EIG_V[i]
    end
    return state
end
end

#=
    The function below is no longer used.
    It has been kept for reference.
=#
function calculate_eigenstates(system::SpinQuantSystem)
    H = system.Hs ⊗ identityoperator(system.Hc) + identityoperator(system.Hs) ⊗ system.Hc + system.V
    HC_EIG_E, HC_EIG_V = eigenstates(dense(system.Hc))
    GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
    PSI_EIG_E, PSI_EIG_V = eigenstates(dense(system.Hs))
    return HC_EIG_E, HC_EIG_V, GLOB_EIG_E, GLOB_EIG_V, PSI_EIG_E, PSI_EIG_V
end
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%