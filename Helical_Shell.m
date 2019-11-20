function average_Hz = Helical_Shell(Field_Str,Noise_Str)
    N = 64;
    mid = round(N/2);
    L = 70; % <-- [pc]
    POINT_SPACING = 2*L/(N-1);

    a = 1.3;
    b = 10;
    V0 = 0.0037; % <-- [pc/year]
    R = 50; % <-- [pc]

    t = 1000; % <-- [years]

    tic
    % Final Eulerian mesh
    [x, y, z] = meshgrid( linspace(-L,L,N), linspace(-L,L,N), linspace(-L,L,N) );
    % final distance r
    r = sqrt(x.^2 + y.^2 + z.^2);
    % initial distance r0
    r0 = arrayfun(@(my_r)initFromFin(my_r,t,a,b,R,V0), r);
    % dr/dr0
    dr_dr0 = arrayfun(@(my_r, my_r0)calc_dr_dr0(my_r,my_r0,a,b,R), r, r0);
    % initial Lagrangian nesh
    x0 = r0.*x./r;

    y0 = r0.*y./r;

    z0 = r0.*z./r;

    if (mod(N,2)==1) % middle point giving nan
        dr_dr0(mid, mid, mid) = 1;
        x0(mid, mid, mid) = 0;
        y0(mid, mid, mid) = 0;
        z0(mid, mid, mid) = 0;
    end

    % derivatives and Jacobian

    dx_dx0 = dr_dr0 .* (x0./r0).^2 + r./r0.*(1-(x0./r0).^2);
    dy_dy0 = dr_dr0 .* (y0./r0).^2 + r./r0.*(1-(y0./r0).^2);
    dz_dz0 = dr_dr0 .* (z0./r0).^2 + r./r0.*(1-(z0./r0).^2);

    dx_dy0 = (dr_dr0 - r./r0) .* x0.*y0./r0.^2;
    dy_dx0 = dx_dy0;

    dx_dz0 = (dr_dr0 - r./r0) .* x0.*z0./r0.^2;
    dz_dx0 = dx_dz0;

    dy_dz0 = (dr_dr0 - r./r0) .* y0.*z0./r0.^2;
    dz_dy0 = dy_dz0;

    inv_J = arrayfun(@(A,B,C,D,E,F,G,H,I)1/det([A,B,C;D,E,F;G,H,I]),dx_dx0,dx_dy0,dx_dz0,dy_dx0,dy_dy0,dy_dz0,dz_dx0,dz_dy0,dz_dz0);


    % Number densities
    n0 = ones(N,N,N);
    n = inv_J .* n0;
    ne = n;
    mean_ne = mean(mean(mean(ne)));
    ngamma = n;
    mean_ngamma = mean(mean(mean(ngamma)));

    % Magnetic field
    Bx0 = Field_Str * ones(N,N,N) / 2^0.5 + Noise_Str * randn(N,N,N) / 3^0.5;
    By0 = Field_Str * cos(pi/L*x) / 2^0.5 + Noise_Str * randn(N,N,N) / 3^0.5;
    Bz0 = Field_Str * sin(pi/L*x) / 2^0.5 + Noise_Str * randn(N,N,N) / 3^0.5;

    Bx = inv_J .* (Bx0.*dx_dx0 + By0.*dx_dy0 + Bz0.*dx_dz0);
    By = inv_J .* (Bx0.*dy_dx0 + By0.*dy_dy0 + Bz0.*dy_dz0);
    Bz = inv_J .* (Bx0.*dz_dx0 + By0.*dz_dy0 + Bz0.*dz_dz0);

    % % Quiver plot of Final Magnetic Field
    % figure();
    % quiver3(x,y,z,Bx,By,Bz);
    % title('Final Magnetic field');
    % xlabel('X / pc');
    % ylabel('Y / pc');
    % zlabel('Z / pc');

    % Initial polarisation angles
    psi0 = mod(atan2(By,Bx) + pi/2, 2*pi);

    % OBSERVABLES
    k = 0.81; % <-- Important quantity [rad / (m^2 cm^-3 microG pc)]
    lambda1 = 0.03; % <-- Wavelength [m]
    lambda2 = 0.06; % <-- Wavelength [m]


    % Intensity

    I = POINT_SPACING * trapz( ngamma.*(Bx.^2 + By.^2) , 3 );
    % Integrate using trapezoidal method
    %  - spacing between points is constant and is equal to POINT_SPACING
    %  - the integrand as given in the formula
    %  - integrating over 3rd axis (i.e. along z-axis)


    % Faraday rotation effect

    F = k * POINT_SPACING * cumtrapz( ne .* Bz , 3 );

    % Final Polarisation angle

    psi1 = psi0 + lambda1^2*F;
    psi2 = psi0 + lambda2^2*F;

    % U and Q, and PI to test that U and Q are components and I is magnitude
    U1 = POINT_SPACING * trapz( ngamma .* (Bx.^2 + By.^2) .* sin(psi1) , 3 );
    Q1 = POINT_SPACING * trapz( ngamma .* (Bx.^2 + By.^2) .* cos(psi1) , 3 );
    PI1 = sqrt(U1.^2 + Q1.^2);

    U2 = POINT_SPACING * trapz( ngamma .* (Bx.^2 + By.^2) .* sin(psi2) , 3 );
    Q2 = POINT_SPACING * trapz( ngamma .* (Bx.^2 + By.^2) .* cos(psi2) , 3 );
    PI2 = sqrt(U2.^2 + Q2.^2);

    % CALCULATIONS FROM OBSERVABLES

    % Final observable (averaged) polarisation angle
    mean_psi1 = mod(atan2(U1,Q1), 2*pi);
    mean_psi2 = mod(atan2(U2,Q2), 2*pi);
    mean_psi = (mean_psi1 + mean_psi2) / 2;

    % Observed (averaged) Faraday's rotation measure
    mean_F = (mean_psi2 - mean_psi1) / (lambda2^2 - lambda1^2);

    % Averaged Bz
    mean_Bz = mean_F / mean_ne / k / (2*L);

    % Averaged Bx and By
    mean_Bxy = sqrt(I / mean_ngamma / (2*L));
    mean_Bx = mean_Bxy .* cos(mean_psi-pi/2);
    mean_By = mean_Bxy .* sin(mean_psi-pi/2);

    % Averaged Jz
    mean_Jz = zeros(N,N);
    for i=2:N-1
        for j=2:N-1
            mean_Jz(i,j) = -sum(mean_Bx(i-1,j-1:j+1)) * ( x(i-1,j-1,end)-x(i-1,j+1,end) );
            mean_Jz(i,j) = mean_Jz(i,j) - sum(mean_Bx(i+1,j-1:j+1)) * ( x(i+1,j+1,end)-x(i+1,j-1,end) );
            mean_Jz(i,j) = mean_Jz(i,j) - sum(mean_By(i-1:i+1,j-1)) * ( y(i+1,j-1,end)-y(i-1,j-1,end) );
            mean_Jz(i,j) = mean_Jz(i,j) - sum(mean_By(i-1:i+1,j+1)) * ( y(i-1,j+1,end)-y(i+1,j+1,end) );
            mean_Jz(i,j) = mean_Jz(i,j) / abs(x(i-1,j-1,end)-x(i-1,j+1,end)) / abs(y(i+1,j-1,end)-y(i-1,j-1,end));
        end
    end

    % Z-component of Helicity -ish
    Hz = mean_Jz .* mean_Bz;
    average_Hz = mean(Hz(:));
    
end

