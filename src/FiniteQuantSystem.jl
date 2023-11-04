using QuantumOptics

abstract type QuantumSystem end

struct SpinQuantSystem <: QuantumSystem
    Hs::Operator
    Hc::Operator
    V::Operator
    H = Hs ⊗ identityoperator(Hc) + identityoperator(Hs) ⊗ Hc + V
    HC_EIG_E, HC_EIG_V = eigenstates(dense(Hc))
    GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
    PSI_EIG_E, PSI_EIG_V = eigenstates(dense(Hs))
    index::nothing
    if index == nothing
        Ψ = PSI_EIG_V[1]
        EΨ = PSI_EIG_E[1]
    else
        Ψ = GLOB_EIG_V[index]
        EΨ = GLOB_EIG_E[index]
    end
    Ψ = GLOB_EIG_V[index]
    EΨ = GLOB_EIG_E[index]
end

function UpdateIndex(System::SpinQuantSystem, index::Int64)
    System.index = index
    System.Ψ = System.GLOB_EIG_V[index]
    System.EΨ = System.GLOB_EIG_E[index]
    println("Index updated to $index")
end



function φ_λ(t::Float64, E::Float64, Ψ::Ket)
    clc_v = Χ(E, t)
    clock_coeff_conj = [dagger(clc_v)*c_basis for c_basis in HC_EIG_V]
    state = PSI_EIG_V[1]*0.0
    for i in eachindex(PSI_EIG_E)
        coeff = 0.0
        for j in eachindex(HC_EIG_E)
            coeff += clock_coeff_conj[j] * dagger(PSI_EIG_V[i] ⊗ HC_EIG_V[j]) * Ψ
        end
        state += coeff * PSI_EIG_V[i]
    end
    return state
end
