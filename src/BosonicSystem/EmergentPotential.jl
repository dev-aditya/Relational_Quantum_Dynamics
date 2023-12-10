include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
PyPlot.rc("axes", grid=true)
using LaTeXStrings
using Base.Threads
using Statistics
using FFTW # for fft
using Peaks
#= 
1. Import the required Hamiltonians 
2. Don't forgert to update the titles and labels of the plots. 
=#
include("hamiltonian/CoupledHarmonicOscillator.jl")
quant_system = BosonQuantumSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;

α = (1 + √5)/2
function χ(t::Float64, E::Float64)
    return coherentstate(bclc, exp(-im * Ω * t)*α)
end
sz = sigmaz(SpinBasis(1//2))
sy = sigmay(SpinBasis(1//2))
sx = sigmax(SpinBasis(1//2))
N_samp = 2^11 - 1
t0 = 0
tmax = π
Ts = tmax / N_samp
# time coordinate
T = t0:Ts:tmax
sqrtxsys = sqrt(dense(xsys).data)
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    local VGLOB = V*projector(quant_system.Ψ)
    VGLOB = VGLOB + dagger(VGLOB)
    local vcomp = Vector{ComplexF64}(undef, length(T))
    local class_coeff = Vector{ComplexF64}(undef, length(T))
    #local vx = Vector{ComplexF64}(undef, length(T))
    #local vy = Vector{ComplexF64}(undef, length(T))
    @threads for i in eachindex(T)
        Χt = χ(T[i], quant_system.EΨ)
        local NORM_ = dagger(quant_system.Ψ)*tensor(identityoperator(Hs), projector(Χt))* quant_system.Ψ
        local V = tensor(identityoperator(Hs), dagger(Χt)) * VGLOB * tensor(identityoperator(Hs), Χt) / λ ## Note the normalization factor
        class_coeff = expect(xclc, Χt)
        sqr_ = sqrt(class_coeff)*sqrtxsys
        if  abs(NORM_) == 0
            println("NORM_ is zero for: ", index)
            vcomp[i] = tr(sqrt(sqr_ * dense(V).data * sqr_))
        else
            V = dense(V / NORM_).data
            vcomp[i] = tr(sqrt(sqr_ * V * sqr_)) 
        end
    end
    ene = quant_system.EΨ
    entan = real(entanglement_entropy(quant_system.Ψ, [2])) / log(2)
    fig, ax = subplots(1, 1, figsize=(10, 6))
    ax.plot(T, real.(vcomp), label= L"Re(Tr[V_{class} * V_{emerg}]) / \lambda")
    ax.set_xlabel(L"t")
    ax.set_ylabel(L"Feidelity")
    ax.set_title("CoupledHarmonicOscillator \n E = $ene, S = $entan")
    ax.legend()
    ax.grid(true)
    fig.savefig("data/V_emerg/emergentPotential_Index_$index.png",)
    close(fig)
end