using QuantumOptics
N_cutoff = 1000

# Define the spin-boson Hamiltonian
spin_basis = SpinBasis(1//2);
clck_basis = FockBasis(N_cutoff);

a = destroy(clck_basis)
ad = create(clck_basis)
ω0 = 2π*1.1
ħ = 1.0
Hs = ħ*ω0*sigmaz(spin_basis)/2
Hc = ħ*ω0*(ad*a)
g = 2π*0.1
V = ħ*g*((sigmap(spin_basis)⊗a) + (sigmam(spin_basis)⊗ad))
