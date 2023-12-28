using QuantumOptics
N_cutoff = 1000

# Define the spin-boson Hamiltonian
spin_basis = SpinBasis(1//2);
clck_basis = FockBasis(N_cutoff);

a = destroy(clck_basis)
ad = create(clck_basis)
ω = 2π*1.0
g = sqrt(10)
Ω = 2π*2.0
ħ = 1.0
g = ħ*g
Hs = ħ*ω*sigmaz(spin_basis)/2
Hc = ħ*Ω*(ad*a + identityoperator(clck_basis)/2)
x  = (a + ad)/√2
V = g*(tensor(sigmax(spin_basis), x))
