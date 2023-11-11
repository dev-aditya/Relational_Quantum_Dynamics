using QuantumOptics

b = SpinBasis(1 // 2);
SIGZ = sigmaz(b);
SIGX = sigmax(b);
SIGY = sigmay(b);
I = identityoperator(b);

N = 8 # Number of spins
# Define the Hamiltonian

function sig(oper::Operator, k::Int64, size::Int64);
    list_ = [sparse(I) for i in 1:size]
    list_[k] = oper
    return tensor(list_...)
end

Hs = 0.0*SIGZ
Hc = 0.0*sig(SIGZ, 1, N - 1)

for i in 1:N-1
    for j in i+1:N-1
        global Hc += sig(SIGX, i, N - 1) * sig(SIGX, j, N - 1) + sig(SIGY, i, N - 1) * sig(SIGY, j, N - 1) + sig(SIGZ, i, N - 1) * sig(SIGZ, j, N - 1)
    end
end
V =  sum([sig(SIGX, 1, N) * sig(SIGX, i, N) + sig(SIGY, 1, N)* sig(SIGY, i, N) + sig(SIGZ, 1, N)*sig(SIGZ, i, N) for i in 2:N])