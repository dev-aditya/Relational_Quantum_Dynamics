using QuantumOptics
using PyPlot

include("hamiltonian/CoupledHarmonicOscillatorX2.jl")
φ = (1 + √5)/2 
α = 300
α = sqrt(α) * exp(im * φ)
H = identityoperator(Hs) ⊗ Hc + Hs ⊗ identityoperator(Hc) + V/abs(α)
GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))
N = identityoperator(Nsys) ⊗ Nclc + Nsys ⊗ identityoperator(Nclc)
Num = abs.(expect(N, GLOB_EIG_V))

# Create a new figure
figure(figsize=(6, 8))

# Plot the 2D histogram
hist2D(GLOB_EIG_E, Num, bins=(300, 300), cmap="plasma", cmin=1)
colorbar(orientation="horizontal")
# Set labels and title
xlabel("Energy")
ylabel(L"\langle N \rangle")
grid(true)
title("CoupledHarmonicOscillator with Xs x Xc^2 coupling at CutOff N = $N_")
PyPlot.savefig("data/NumberVsEnergy.png", dpi=300)
println("done")
close("all")