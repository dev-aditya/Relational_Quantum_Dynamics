include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Plots
using Base.Threads
using Statistics
using LaTeXStrings
using DelimitedFiles

include("hamiltonian/powerLawCoupling.jl")

quant_system = SpinQuantSystem(Hs, Hc, V);
HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;
function χ(t::Float64, E::Float64)
    return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
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

T = Base._linspace(0.0, 2 * π, 1000);
var_c1 = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
var_c2 = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
ent_vec = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
for index in eachindex(quant_system.GLOB_EIG_E)
    UpdateIndex(quant_system, index)
    global c1 = Vector{ComplexF64}(undef, length(T))
    global c2 = Vector{ComplexF64}(undef, length(T))

    @threads for i in eachindex(T)
        t = T[i]
        ϕ = φ_λ(t, quant_system)
        ϕ = ϕ / norm(ϕ)
        c1[i] = ϕ.data[1]
        c2[i] = ϕ.data[2]
    end
    #c1 = abs.(c1)
    #c2 = abs.(c2)
    #var_c1[index] = var(c1)
    #var_c2[index] = var(c2)
    entan = real(entanglement_entropy(quant_system.Ψ, [i for i=2:N]))/log(2)
    ent_vec[index] = entan
    plot(T, abs.(c1), label=("c1 Ene: " * string(quant_system.EΨ)), legend=:right)
    plot!(T, abs.(c2), label=("c2 Ent: " * string(entan)), legend=:right, figsize=(19.20, 15.80))
    xlabel!("t")
    ylabel!(L"|c_1|^2, |c_2|^2")
    title!("Power Law Coupling of N = $N spins, " * L"\gamma: " * "$γ", fontsize=9)
    savefig("data/index_$index" * "powerlaw_coupling" * ".png",)
end
histogram2d(
    quant_system.GLOB_EIG_E,  
    ent_vec, bins=(100, 100), 
    show_empty_bins=false, 
    color=:plasma, 
    xlabel="Energy",
    xticks=-5:1:20,
    ylabel=L"Entropy", 
    fontsize=9)
savefig("data/Entropy_Energy_for_N=$N _powerlaw_coupling.png")
#= p1 = histogram2d(
    quant_system.GLOB_EIG_E,  
    var_c1, bins=(70, 70), 
    show_empty_bins=false, 
    color=:plasma, 
    xlabel="Energy",
    ylabel=L"\sigma^2_{c1}", 
    fontsize=9)
p2 = histogram2d(
    quant_system.GLOB_EIG_E,  
    var_c2, bins=(70, 70), 
    show_empty_bins=false, 
    color=:plasma, 
    xlabel="Energy",
    ylabel=L"\sigma^2_{c2}", 
    fontsize=9)
l = @layout [ a b ]
plot(p1, p2, layout=l, size=(700, 350))
savefig("data/Variance of abs(coeff) for N = $N and g = 1.png") =#
