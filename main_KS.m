N = 128;
L = 22;

x = (0:N-1)/N*2*pi;
x = x';

%initial data
u = cos(2*x-1) + 1 - sin(x);

%perturbations
v = [0*x+1, cos(x), sin(x)];

T = 10;
steps = 128;

% FIrst test runtime with size of v
ms = 1:32;
trials = 5;
times = zeros( trials, numel(ms) );
for m = ms
  m;
  for t = 1:trials
  tic
  v = rand(N,m);
  visualize = false;
  [u,~] = forward_shooting( u, v, T, steps, L, visualize );
  times(t,m) = toc;
  end
end


%%
clf;
errorbar(ms, mean(times), std(times), 'color', 'blue', 'linewidth', 2 );
xlabel("block size");
ylabel("walltime of matrix multiplication (s)");
title("Batch Jacobian evaluation of KS");
pbaspect([2,1,1]);
yticks([1:3] * 0.01);
%% Newton-Krylov iteration

%initial data
L = 22;
u = cos(x) - sin(2*x-2) - 0.2;

%spatial perturbations to use in X0
kmax = 3;
v = [0*x+1, cos(x.*(1:kmax)), sin(x.*kmax) ];
m = size(v, 2);

T = 20;
steps = 256;

z  = [u;T];

figure(4)
U = zeros(N,steps+1);
U(:,1) = z(1:N);
for i = 2:steps+1
  visualize = false;
  U(:,i) = forward_shooting( U(:,i-1), 0*U(:,1), z(end)/steps, 1, L, visualize );
end
imagesc(U);
colorbar();
drawnow;



%Initial guess for block_GMRES
X0 = zeros(N+1, m+1);
X0(1:N,1:m) = v;
X0(  N,m+1) = 1; %add just a shift in period

%Newton parameters
maxit = 128;
hook  = 0.1;
inner = 4;
outer = 1;

norm_f     = zeros(maxit, 1);
bGMRES_res = zeros(maxit, 1);

for i = 1:maxit
  %compute f and define an anonymous function for Jacobian eval
  [~, f]  = PO_objective( z, 0*z, L, steps, N );
  J = @(dz) PO_objective( z, dz, L, steps, N );

  % block_gmres evaluation
  dz = block_gmres( J, f, X0, inner, outer );
  
  %update Newton info
  norm_f(i)     = norm(f);
  bGMRES_res(i) = norm(J(dz) - f)/norm(f);
  fprintf("Step %d: |f| = %e\n", i, norm(f) );

  z = z - hook*dz;
end

visualize_newton_output( z, L, steps, norm_f, bGMRES_res );





function visualize_newton_output( z, L, steps, norm_f, bGMRES_res )
  maxit = numel(norm_f);
  N = size(z,1)-1;
  T = z(end);

  figure(2);
  semilogy( 1:maxit, norm_f );
  ylabel("|f|"); 
  xlabel("Newton iteration");

  figure(3);
  plot( 1:maxit, bGMRES_res );
  ylabel("bGMRES residual");
  xlabel("Newton iteration");

  figure(4)
  U  = zeros(N,steps+1);
  U(:,1) = z(1:N);
  for i = 2:steps+1
    visualize = false;
    U(:,i) = forward_shooting( U(:,i-1), 0*U(:,1), T/steps, 1, L, visualize );
  end
  surf(U);
  shading interp;
  colorbar();
end

function [Jv, f] = PO_objective( z, dz, L, steps, N )
  %{
  Compute a periodic orbit cost and the action of the Jacobian
  %}
  visualize = false;
  u0 =  z(1:N);
  v0 = dz(1:N,:);
  [u,v] = forward_shooting( u0, v0, z(end), steps, L, visualize );
  
  f      = zeros(N+1,1);
  f(1:N) = u - u0;

  k = 0:N-1; k(k>N/2) = k(k>N/2) - N;
  k = k' / L * 2 * pi;
  dudx = real(ifft(1i*k.*fft(u0))); %x deriv of INITIAL u
  dudt = real(ifft(-0.5*1i*k.*fft(u.^2) + k.^2 .*fft(u) - k.^4.*fft(u) )); % time deriv of FINAL u

  m  = size(dz,2);
  Jv = zeros(N+1, m);
  Jv(1:N,:) = v - v0 + dudt .* dz(end,:);
  Jv(N+1,:) = sum(v.*dudx); %phase condition. Fix translation in x.
end

function [u,v] = forward_shooting( u, v, T, steps, L, visualize )
  %{
  PURPOSE:
  Integrate an initial condition u forward in time, and integrate a set of
  perturbations v forward under the linearized dynamics.
  %}

  N = numel(u);
  k = 0:N-1; k(k>N/2) = k(k>N/2) - N;
  k = k' / L * 2 * pi;

  dt = T/steps;
  diss = exp( dt/2 * (k.^2 - k.^4) ); %half step in dissipation
  mask = abs(k) < N/3;

  %Macro for applying dissipation over dt/2
  diss_step = @(u) real(ifft( mask.*diss.*fft(u) ));

  for i = 1:steps
    %Half step in dissipation
    u = diss_step(u);
    v = diss_step(v);

    %Full step in advection (RK2)
    [k1, K1] = advection(u,           v,           k);
    [k2, K2] = advection(u + dt*k1/2, v + dt*K1/2, k);
    u = u + dt*k2;
    v = v + dt*K2;

    %Half step in dissipation
    u = diss_step(u);
    v = diss_step(v);

    if visualize
      clf
      tiledlayout(1,2);
      nexttile;
      plot(u);
      title("u");
      ylim([-3 3]);

      nexttile
      plot(v ./ vecnorm(v));
      title("normalized v");
      ylim([-1 1]);
      drawnow;
    end
  end
end

function [f, Jv] = advection(u, v, k)
  % f = -u*dudx

  dudx = real(ifft( 1i*k.*fft(u)));
  dvdx = real(ifft( 1i*k.*fft(v)));

  f  = -u.*dudx;
  Jv = -v.*dudx - u.*dvdx;
end