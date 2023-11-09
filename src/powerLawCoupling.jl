using QuantumOptics

b = SpinBasis(1 // 2);
SIGZ = sigmaz(b);
SIGX = sigmax(b);
SIGY = sigmay(b);
I = identityoperator(b);

N = 8 # Number of spins
# Define the Hamiltonian
γ = 2.0
l = 1.0

function sig(oper::Operator, k::Int64, size::Int64);
    list_ = [sparse(I) for i in 1:size]
    list_[k] = oper
    return tensor(list_...)
end

## --- Hamiltonian for pure spin-spin interaction with external field --- ##
Hs = 0.0*SIGZ;
Hc = 0.0*sum([sig(SIGZ, i, N - 1) for i in 1:N-1]);

for i in 1:N-1
    for j in i+1:N-1
        global Hc += 1/abs(l*i - l*j)^γ * sig(SIGX, i, N - 1) * sig(SIGX, j, N - 1);
    end
end
V = 1/abs(l)^γ * sig(SIGX, 1, N) * sum([sig(SIGX, i, N) for i in 2:N]);