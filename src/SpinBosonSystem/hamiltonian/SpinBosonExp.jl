using QuantumOptics
N_cutoff = 2000

# Define the spin-boson Hamiltonian
spin_basis = SpinBasis(1//2);
clck_basis = FockBasis(N_cutoff);

a = destroy(clck_basis)
ad = create(clck_basis)
ω = 2π*0.1
g = 1000.0
Ω = 2π*2.0
ħ = 1.0
g = ħ*g
Hs = ħ*ω*sigmaz(spin_basis)/2
Hc = ħ*Ω*(ad*a + identityoperator(clck_basis)/2)
x  = (a + ad)/√2
σ = √1
#V = g*(tensor(sigmax(spin_basis), exp(x^2/σ^2)))