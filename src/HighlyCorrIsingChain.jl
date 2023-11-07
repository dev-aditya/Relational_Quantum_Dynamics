include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using Plots
using Base.Threads
using Statistics

b = SpinBasis(1 // 2)
SIGZ = sigmaz(b)
SIGX = sigmax(b)
I = identityoperator(b)

N = 4 # Number of spins
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
var_c1 = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
var_c2 = Vector{Float64}(undef, length(quant_system.GLOB_EIG_E))
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
    c1 = abs.(c1)
    c2 = abs.(c2)
    var_c1[index] = var(c1)
    var_c2[index] = var(c2)
    #entan = real(entanglement_entropy(quant_system.Ψ, [i for i=2:N]))/log(2)
    #plot(T, abs.(c1), label=("Energy: " * string(quant_system.EΨ)), legend=:right)
    #plot!(T, abs.(c2), label=("Entangement: " * string(entan)), legend=:right, figsize=(19.20, 15.80))
    #xlabel!("t")
    #ylabel!("|c1|^2, |c2|^2")
    #title!("Time evolution of the states of the system for Index: " * string(index))
    #savefig("data/Time evolution of thestates of the system for Index: " * string(index) * ".png",)
end
p1 = histogram2d(
    quant_system.GLOB_EIG_E,  
    var_c1, bins=(70, 70), 
    show_empty_bins=false, 
    color=:plasma, 
    title="Variance of c1",
    xlabel="Energy",
    ylabel="Variance of c1")
p2 = histogram2d(
    quant_system.GLOB_EIG_E,  
    var_c2, bins=(70, 70), 
    show_empty_bins=false, 
    color=:plasma, 
    title="Variance of c2",
    xlabel="Energy",
    ylabel="Variance of c2")
plot(p1, p2, layout=(1, 2), title = "Coeff Variance VS Energy for N = $N and g = 1")
savefig("data/Variance of abs(coeff) for N = $N and g = 1.png")