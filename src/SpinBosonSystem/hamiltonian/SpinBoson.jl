using QuantumOptics
N_cutoff = 500

# Define the spin-boson Hamiltonian
spin_basis = SpinBasis(1//2);
clck_basis = FockBasis(N_cutoff);

a = destroy(clck_basis)
ad = create(clck_basis)
ω = 2π*0.1
g = sqrt(2)
Ω = 2π*2.0
ħ = 1.0
g = ħ*g
Hs = ħ*ω*sigmaz(spin_basis)/2
Hc = ħ*Ω*(ad*a + identityoperator(clck_basis)/2)
V = g*(tensor(sigmax(spin_basis), (a + ad)/√2))
