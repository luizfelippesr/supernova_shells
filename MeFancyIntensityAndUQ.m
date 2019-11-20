function MeFancyIntensityAndUQ(x, y, I, U,Q, each)

figure1 = figure;
axes1 = axes('Parent', figure1);
hold(axes1,'on');

surf(x(:,:,end),y(:,:,end),I,'Parent',axes1,'EdgeColor','none');
colorbar();

max_I = max(max(I));
max_I = max_I + 10;
z = ones(size(U)) * max_I;
w = zeros(size(U));

quiver3(x(1:each:end,1:each:end,end),y(1:each:end,1:each:end,end),z(1:each:end,1:each:end),U(1:each:end,1:each:end), Q(1:each:end,1:each:end), w(1:each:end,1:each:end),'Parent',axes1,'Color',[0 0 0], 'ShowArrowHead', 'off','LineWidth',1);
quiver3(x(1:each:end,1:each:end,end),y(1:each:end,1:each:end,end),z(1:each:end,1:each:end),-U(1:each:end,1:each:end), -Q(1:each:end,1:each:end), w(1:each:end,1:each:end),'Parent',axes1,'Color',[0 0 0], 'ShowArrowHead', 'off','LineWidth',1);



grid(axes1,'on');
axis(axes1,'tight');
% Set the remaining axes properties
set(axes1,'DataAspectRatio',[1 1 1],'FontSize',20);

% Create ylabel
ylabel('$y$ / pc','FontSize',24,'Interpreter','latex');

% Create xlabel
xlabel('$x$ / pc','FontSize',24,'Interpreter','latex');
