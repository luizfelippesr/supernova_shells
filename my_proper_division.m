function value = my_proper_division(r,r0)
%my_proper_division
%   this function takes into the account that when both r and r0 are 0
%   their ration must be 1
if (r0 == 0 && r == 0)
    value = 1;
else
    value = r/r0;
end
end

