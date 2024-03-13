function [x] = block_gmres(A, b, X0, inner, outer)
  
  n = size(X0, 1);
  m = size(X0, 2) + 1;
  
  %approximately Hessenberg matrix
  H = zeros( m*(inner+1), m*inner );
  Q = zeros( n, m*inner); %orthonormal basis from power iteration
  x = zeros( n, 1); %solution vector we will build iteratively

  for j = 1:outer
    X0_mod = [X0, b]; %always append the rhs to block guess so outer iteration is meaningful
    [Q(:,1:m), ~] = qr( X0_mod, "econ" ); %find orthonormal basis of X0 with QR decomposition
    for i = 1:inner
      %Do a loop over inner iterations
      current_block = (i-1)*m + (1:m);
      next_block    = (i  )*m + (1:m);
      past_blocks   = 1:i*m;

      %Evaluate on the current block
      Aq = A(Q( :, current_block )); %replace with matrix multiplication
      H( past_blocks, current_block ) = Q( :, past_blocks )' * Aq; %Project onto past basis
      Aq = Aq - Q(:,past_blocks)*H(past_blocks, current_block); %Orthogonalize with respect to this basis
      [Q(:,next_block), H(next_block, current_block)] = qr( Aq, "econ" );
    end
  
    b2 = Q'*b; %Project b into Krylov subspace
    tol = 1e-2;
    x2 = pinv(H, tol) * b2;
    x_outer = Q( :, 1:inner*m )*x2; %x guess from this outer iteration
    x = x + x_outer;
    b = b - A(x_outer);
  end
  
  %{
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
  %}

  %residual = norm( A(x) - b) / norm(b);
  %fprintf("residual = %e\n", residual );
end