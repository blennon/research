N = 5;
M = magic(N); % 5x5 matrix

M_diag = zeros(N,1);
for i=1:N
    E = zeros(N,N);
    e = zeros(N,1);
    e(i,1) = 1;
    E(i,i) = 1;
    M_diag = M_diag + E*M*e;%(e'*(M'*E))';
end
diag(M)
M_diag
diag(M) - M_diag
