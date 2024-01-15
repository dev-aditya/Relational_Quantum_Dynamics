include("FiniteQuantSystem.jl")
using .FiniteQuantSystem
using QuantumOptics
using PyPlot
PyPlot.rc("axes", grid=true)
using LaTeXStrings
using Base.Threads
using Statistics
using Peaks


b = SpinBasis(1 // 2);
SIGZ = sigmaz(b);
SIGX = sigmax(b);
SIGY = sigmay(b);
I = identityoperator(b);

# Define the Hamiltonian
γ = 2.0
l = 1.0

function sig(oper::Operator, k::Int64, size::Int64);
    list_ = [sparse(I) for i in 1:size]
    list_[k] = oper
    return tensor(list_...)
end

Ket0 = spinup(b)
Ket1 = spindown(b)
N_samp = 2^11 - 1
t0 = 0
tmax = pi
Ts = tmax / N_samp
# time coordinate
global T = t0:Ts:tmax
Nlis = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11,]
global rt_avg = zeros(Float64, length(Nlis))
for j in eachindex(Nlis)
    N = Nlis[j]
    ## --- Hamiltonian for pure spin-spin interaction with external field --- ##
    local Hs = 1.0*SIGZ;
    local Hc = 1.0*sum([sig(SIGZ, i, N - 1) for i in 1:N-1]);

    for i in 1:N-1
        for j in i+1:N-1
            distance = min(abs(i - j), N - abs(i - j))
            Hc += 1/abs(l*distance)^γ * sig(SIGX, i, N - 1) * sig(SIGX, j, N - 1);
        end
    end
    local V = sig(SIGX, 1, N) * sum([1/abs(l*min(abs(1 - i), N - abs(1 - i)))^γ * sig(SIGX, i, N) for i in 2:N])

    local quant_system = SpinQuantSystem(Hs, Hc, V);
    HC_EIG_E_loc, HC_EIG_V_loc = quant_system.HC_EIG_E, quant_system.HC_EIG_V;

    function χ(t::Float64, E::Float64)
        return sum([exp(-im * (HC_EIG_E_loc[i] - E) * t) * HC_EIG_V_loc[i] for i in eachindex(HC_EIG_E_loc)])
    end

    local r_t = Vector{ComplexF64}(undef, length(T))
    local Φ = Vector{Ket}(undef, length(T))
    for i in eachindex(T)
        Φ[i] = normalize(tensor(identityoperator(Hs), dagger(χ(T[i], quant_system.EΨ))) * quant_system.Ψ)
        r_t[i] = (dm(Φ[i]).data)[1, 2] / (dm(Φ[1]).data)[1, 2]
    end
    rt_avg[j] = mean(abs2.(r_t))
    entan = real(entanglement_entropy(quant_system.Ψ, [i for i = 2:N])) / log(2)
    println("N: $(Nlis[j])")
    println("Energy: $(quant_system.EΨ)")
    println("Entanglement: $(entan)")
    println("------------------------------------")
end

figure(figsize=(15, 10))
plot(Nlis, rt_avg, marker="o", linewidth=1.0)
xlabel(L"N")
ylabel(L"<|r(t)|^2>")
title("Decoherence vs N")
grid()
savefig("data/DecoherenceVsN.png")
