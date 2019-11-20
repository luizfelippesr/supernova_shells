function r = finFromInit(r0, t, a, b, R, V0)
%Final (Eulerian) radius from Initial (Lagrangian)
%   This function returns you r from initial radius, r0, 
%   time, t, and given parameters of the field: a, b, R, V0
%   a != 1 and b != 0

% If we start inside the shell
if (r0 < R)
    % Time it takes to reach from r0 to the R is Tau
    Tau = R/(1-a)/V0 * ( 1 - (r0/R)^(1-a) );
    
    % if Tau is more than t, then it means that it takes more time to reach
    % parameter radius R than time to reach final position r
    % In other words, r<R
    if (Tau >= t)
        r = ( r0^(1-a) + (1-a)*V0*t/R^a )^(1/(1-a));
        
    % if Tau is less than t, then it means that it takes less time to reach
    % R than r
    % In other words, r>=R
    % Here, we start within R, but finish otside
    else
        r = R * ( 1 + log( 1 + b*V0*t/R - b*(1 - (r0/R)^(1-a))/(1-a) )/b );
    end
    
% If we start outside the shell
else
    r = R * ( 1 + log(exp(b*(r0/R-1)) + b/R*V0*t)/b );
end

end

