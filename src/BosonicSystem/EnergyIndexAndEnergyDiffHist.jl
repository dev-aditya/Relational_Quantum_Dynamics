using QuantumOptics
using PyPlot

include("hamiltonian/CoupledHarmonicOscillator.jl")
H = identityoperator(Hs) ⊗ Hc + Hs ⊗ identityoperator(Hc) + V
GLOB_EIG_E, GLOB_EIG_V = eigenstates(dense(H))

# Create a new figure
fig, axs = subplots(1, 2, figsize=(20, 10),)

# Plot the 2D histogram
axs[1].plot(GLOB_EIG_E, linewidth=0.5, color="black", marker="o", markersize=0.5,)
axs[1].set_xlabel("index")
axs[1].set_ylabel("Energy")
axs[1].set_title("Energy vs Index")
axs[1].grid(true)

axs[2].hist(diff(GLOB_EIG_E), bins=1000, edgecolor="black", linewidth=0.5, density=false,)
axs[2].set_xlabel(L"\Delta E")
axs[2].set_ylabel(L"d(\Delta E)")
axs[2].set_title("  UnNormalized Energy Difference Histogram")
axs[2].set_xlim(0, 1.0)
axs[2].grid(true)

fig.suptitle("Coupled Harmonic Oscillators")
PyPlot.savefig("data/EnergyDistHist.png")