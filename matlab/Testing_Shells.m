M = 25;
Field = linspace(1,0.2,M);
Noise = 1 - Field;

ratio = zeros(1,M);
Helical = zeros(5,M);
Linear = zeros(5,M);

for i=1:M
    disp(i);
    ratio(i) = Noise(i)/(Noise(i) + Field(i));
    for j=1:5
        disp(j);
        Linear(j,i) = Linear_Shell(Field(i), Noise(i));
        Helical(j,i) = Helical_Shell(Field(i), Noise(i));
    end
end

av_Linear = mean(Linear);
av_Helical = mean(Helical);
std_Linear = std(Linear);
std_Helical = std(Helical);

figure();
hold on;
errorbar(ratio,av_Linear,std_Linear);
errorbar(ratio,av_Helical,std_Helical);
xlabel('Noise To Signal Ratio');
ylabel('- average of Hz');
legend('Linear', 'Helical');