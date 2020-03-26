function value = calc_dr_dr0(r,r0,a,b,R)
%dr/dr0 self-explanatory 

% If we start inside the shell
if (r0 < R) 
    % If we finish inside the shell
    if (r<=R)
        value = (r/r0)^a;
    % If we finish outside the shell 
    else
        value = exp(b*(1-r/R))*(R/r0)^a;
    end

% If we start outside
else
    value = exp(b*(r0-r)/R);
end

end

