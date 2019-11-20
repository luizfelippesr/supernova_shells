function closest = chooseClosest( target, src )
    diff = (target - src);
    temp = fix(2*diff / pi) * pi / 2 + src + [-pi/2, 0, pi/2];
    diff_with_temp = abs(target - temp);
    choice = min(diff_with_temp) == diff_with_temp;
    closest = temp(choice);
    
end

