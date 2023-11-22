using QuantumOptics

bsys = FockBasis(2)
bclc = FockBasis(500)

asys = destroy(bsys)
aclc = destroy(bclc)

Nsys = number(asys)
Nclc = number(aclc)

ω = 2π*1e2
Ω = 2π
ħ = 1.0
Hs = ω*(Nsys + 0.5)
Hc = Ω*(Nclc + 0.5)

V = 2π*1e-3*(asys + dagger(asys)) ⊗ (aclc + dagger(aclc))