%{
This is a simple example of using block_GMRES
%}

rng(1); %seed random number

n = 256;
A = 2*rand(n,n) - 1;
b = 2*rand(n,1) - 1;

m = 3; %block size
X0 = 2*rand( n,m )-1;
X0 = []; %pass nothing for defualt gmres

tol   = 1e-2;
inner = 20; %block evaluations
outer = 3; 
Af = @(v) A*v; %define a function handle for matrix multiplication

[x1,       res1] = block_gmres(Af, b, X0, tol, inner, outer);
[x2, flag, res2] = gmres(       Af,b, inner, tol, outer);

[res1, res2]


