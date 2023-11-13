using QuantumOptics
using PyPlot

include("hamiltonian/powerLawCoupling.jl")
H = identityoperator(Hs) ⊗ Hc + Hs ⊗ identityoperator(Hc) + V
GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
function Entropy(Ψ::Ket)
    return real(entanglement_entropy(Ψ, [i for i=2:N]))/log(2)
end

ent = Entropy.(GLOB_EIG_V)

# Create a new figure
figure(figsize=(6, 8))

# Plot the 2D histogram
hist2D(GLOB_EIG_E, ent, bins=(80, 80), cmap="plasma", cmin=1)
colorbar(orientation="horizontal")
# Set labels and title
xlabel("Energy")
ylabel("Entropy")
title("Entropy Energy for N = $N; gamma = $γ")
PyPlot.savefig("data/EntropyEnergy_powerLawCoupling_$N-spins.png")