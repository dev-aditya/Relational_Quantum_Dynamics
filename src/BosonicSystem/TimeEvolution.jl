include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Plots
using Base.Threads
using Statistics
using LaTeXStrings
using DelimitedFiles

include("hamiltonian/CoupledHarmonicOscillator.jl")

quant_system = BosonQuantumSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;
α = (1 + √5)/2
function χ(t::Float64, E::Float64)
    return coherentstate(bclc, exp(-im * Ω * t)*α)
end

## Calculate Degeneracy
energy_levels = quant_system.GLOB_EIG_E
energy_levels = round.(energy_levels, digits=8)
degeneracy = Dict{Float64, Int}()
for E in energy_levels
    if haskey(degeneracy, E)
        degeneracy[E] += 1
    else
        degeneracy[E] = 1
    end
end

outfile = "data/energy_degeneracy.txt"
writedlm(outfile, degeneracy, '\t')

T = Base._linspace(0.0, 1.0, 1000);
var_c1 = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
var_c2 = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
ent_vec = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    global c1 = Vector{ComplexF64}(undef, length(T))
    global c2 = Vector{ComplexF64}(undef, length(T))

    @threads for i in eachindex(T)
        t = T[i]
        ϕ = tensor(identityoperator(Hs), dagger(χ(t, quant_system.EΨ))) * quant_system.Ψ
        ϕ = ϕ / norm(ϕ)
        c1[i] = ϕ.data[1]
        c2[i] = ϕ.data[2]
    end
    #c1 = abs.(c1)
    #c2 = abs.(c2)
    #var_c1[index] = var(c1)
    #var_c2[index] = var(c2)
    entan = real(entanglement_entropy(quant_system.Ψ, [2]))/log(2)
    ent_vec[index] = entan
    plot(T, abs.(c1), label=("c1 Ene: " * string(quant_system.EΨ)), legend=:right)
    plot!(T, abs.(c2), label=("c2 Ent: " * string(entan)), legend=:right, figsize=(19.20, 15.80))
    xlabel!("t")
    ylabel!(L"|c_1|^2, |c_2|^2")
    title!("CoupledHarmonicOscillator, ", fontsize=9)
    savefig("data/index_$index" * "CoupledHarmonicOscillator" * ".png",)
end
