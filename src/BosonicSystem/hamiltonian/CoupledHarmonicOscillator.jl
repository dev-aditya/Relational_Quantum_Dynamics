using QuantumOptics

bsys = FockBasis(2)
bclc = FockBasis(500)

asys = destroy(bsys)
aclc = destroy(bclc)

Nsys = number(bsys)
Nclc = number(bclc)

ω = 2π
Ω = 2π
ħ = 1.0
Hs = ħ*ω*(Nsys + 0.5*identityoperator(bsys))
Hc = ħ*Ω*(Nclc + 0.5*identityoperator(bclc))

V = ħ*1*(asys + dagger(asys)) ⊗ (aclc + dagger(aclc))