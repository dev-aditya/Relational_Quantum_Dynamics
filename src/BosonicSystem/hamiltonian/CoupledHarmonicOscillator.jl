using QuantumOptics

bsys = FockBasis(1)
bclc = FockBasis(500)

asys = destroy(bsys)
aclc = destroy(bclc)

Nsys = number(bsys)
Nclc = number(bclc)

ω = 2π
Ω = 2π*2.0
ħ = 1.0
Hs = ħ*ω*(Nsys + 0.5*identityoperator(bsys))
Hc = ħ*Ω*(Nclc + 0.5*identityoperator(bclc))
xsys = (asys + dagger(asys))/√2
xclc = (aclc + dagger(aclc))/√2
λ = ħ*√3
V = λ*xsys ⊗ xclc
sqrtxsys = sqrt(dense(xsys).data)