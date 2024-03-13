
rng(1); %seed random number

n = 256;
A = 2*rand(n,n) - 1;

b = 2*rand(n,1) - 1;

m = 16; %block size
X0 = 2*rand( n,m )-1;

inner = 4; %block evaluations
outer = 1; 
[x,residual, H, Q] = block_gmres(A, b, X0, inner, outer);

% Test that I compute what I think I am


[x, ~] = gmres(A, b, m*inner, 1e-9);
gmres_residual = norm(A*x - b)/norm(b)

Q_small = Q(:, 1:m*inner);

imagesc( A*Q_small - Q*H )
colorbar();

imagesc( Q'*Q )

function [x, residual, H, Q] = block_gmres(A, b, X0, inner, outer)
  %{
  Solve A*x = b with Krylov iteration
  %}

  n = size(X0, 1);
  m = size(X0, 2);
  
  %approximately Hessenberg matrix
  H = zeros( m*(inner+1), m*inner );
  size(H)
  Q = zeros( n, m*inner); %orthonormal basis from power iteration

  [Q(:,1:m), ~] = qr( X0, "econ" ); %find orthonormal basis of X0 with QR decomposition
  for i = 1:inner
    %Do a loop over inner iterations
    current_block = (i-1)*m + (1:m);
    next_block    = (i  )*m + (1:m);
    
    past_blocks   = 1:i*m;

    %Evaluate on the current block
    Aq = A*Q( :, current_block );

    %Project onto past basis
    H( past_blocks, current_block ) = Q( :, past_blocks )' * Aq;

    %Orthogonalize with respect to this basis
    Aq = Aq - Q(:,past_blocks)*H(past_blocks, current_block);

    [Q(:,next_block), H(next_block, current_block)] = qr( Aq, "econ" );
  end
  
  b2 = Q'*b; %rotate b
  x2 = lsqr(H, b2);
  x = Q( :, 1:inner*m )*x2;

  tiledlayout(1,2);
  
  nexttile
  imagesc(Q);
  title("Arnoldi basis");

  nexttile
  imagesc(H);
  title("Hessenberg");
  pbaspect([ size(H,2), size(H,1), 1 ])

  for i = 1:m
    xline(i*m);
    yline(i*m);
  end

  residual = norm(A*x - b) / norm(b);
  fprintf("residual = %e\n", residual );
end