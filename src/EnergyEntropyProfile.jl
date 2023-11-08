using QuantumOptics
using Plots

b = SpinBasis(1 // 2)
SIGZ = sigmaz(b)
SIGX = sigmax(b)
SIGY = sigmay(b)
I = identityoperator(b)

N = 12 # Number of spins
# Define the Hamiltonian

function sig(oper::Operator, k::Int64, size::Int64)
    list_ = [sparse(I) for i in 1:size]
    list_[k] = oper
    return tensor(list_...)
end

## --- Hamiltonian for pure spin-spin interaction with external field --- ##
#= Hs = 0.0*SIGZ
Hc = 0.0*sum([sig(SIGZ, i, N - 1) for i in 1:N-1])

for i in 1:N-1
    for j in i+1:N-1
        global Hc += sig(SIGX, i, N - 1) * sig(SIGX, j, N - 1)
    end
end
V = sig(SIGX, 1, N) * sum([sig(SIGX, i, N) for i in 2:N]) =#

## --- Hamiltonian for  spin-spin interaction along x-axis with external field --- ##
Hs = 0.0*SIGZ
Hc = 0.0*sig(SIGZ, 1, N - 1)

for i in 1:N-1
    for j in i+1:N-1
        global Hc += sig(SIGX, i, N - 1) * sig(SIGX, j, N - 1) + sig(SIGY, i, N - 1) * sig(SIGY, j, N - 1) + sig(SIGZ, i, N - 1) * sig(SIGZ, j, N - 1)
    end
end
V =  sum([sig(SIGX, 1, N) * sig(SIGX, i, N) + sig(SIGY, 1, N)* sig(SIGY, i, N) + sig(SIGZ, 1, N)*sig(SIGZ, i, N) for i in 2:N])

H = Hs ⊗ identityoperator(Hc) + identityoperator(Hs) ⊗ Hc + V
GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
function Entropy(Ψ::Ket)
    return real(entanglement_entropy(Ψ, [i for i=2:N]))/log(2)
end

ent = Entropy.(GLOB_EIG_V)

p = histogram2d(
    GLOB_EIG_E, 
    ent, 
    bin=(70, 70),
    xlabel="Energy",
    ylabel="Entropy",
    colormap=:plasma,
    show_empty_bins=false,
    title="Entropy Energy for N = $N"
);
display(p)