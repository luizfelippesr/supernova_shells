function r0 = initFromFin(r, t, a, b, R, V0)
%Initial (Lagrangian) radius from Final (Eulerian)
%   Look at finFromInit function

% If we finish inside the shell
if (r<=R)
    r0 = ( r^(1-a) - (1-a)*V0*t/R^a )^(1/(1-a));
    
% If we finish outside the shell    
else
    % Time it takes to go from R to r is Tau
    Tau = R*( exp(b*(r/R-1)) - 1 )/b/V0;
    
    % If Tau is bigger than t, then we started from outside of the shell
    % In other words r0 >= R
    if (Tau >= t)
        r0 = R * ( 1 + log(exp(b*(r/R-1)) - b/R*V0*t)/b );
        
    % If Tau is smaller than t, then we started from inside the shell
    % and then moved outside
    % In other words r0 < R
    else
        r0 = R*( 1 + (1-a)*( exp(b*(r/R-1)) - 1 - b*V0*t/R )/b )^(1/(1-a));
    end
end

end

