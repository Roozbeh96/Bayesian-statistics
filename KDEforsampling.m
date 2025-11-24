clc; clear; close all;

%% Step 1: Generate Synthetic Data (Example X, Y)
n_samples = 100000; 
rho_target = 0.6; % Target correlation

X = 3 * randn(n_samples, 1) + 5;  % X ~ N(5,3)
N = randn(n_samples, 1);          % Standard normal noise

sigma_X = std(X);
sigma_Y = 1; 

mu_X = mean(X);
mu_Y = 0;
% Solve for b given a = 2
a = rho_target*sigma_Y/sigma_X;
b = sigma_Y*sqrt(1-rho_target^2);

% Generate Y
Y_prime = a * (X-mu_X) + b * N;
Y = a * (X-mu_X) + b * N + mu_Y;

corrcoef(X,Y)
% Plot Joint Density
figure;
scatter(X, Y, 10, 'filled'); hold on;
xlabel('X'); ylabel('Y');
title('Scatter Plot of X and Y');

%% Step 2: Estimate Joint Density p(X,Y) using KDE
% Use 2D kernel density estimation
[f_joint, Xi] = ksdensity([X Y], 'Bandwidth', [0.5, 0.5]); 

figure;
[X_grid, Y_grid] = meshgrid(linspace(min(X), max(X), 50), linspace(min(Y), max(Y), 50));
F = mvksdensity([X, Y], [X_grid(:), Y_grid(:)], 'Bandwidth', [0.5, 0.5]);
F = reshape(F, size(X_grid));

surf(X_grid, Y_grid, F, 'EdgeColor', 'none');
xlabel('X'); ylabel('Y'); zlabel('p(X,Y)');
title('3D Surface of Joint Density p(X, Y)');
colorbar;
grid on;

%% Step 3: Compute Marginal Density p(Y)
[f_Y, Y_vals] = ksdensity(Y, 'Bandwidth', 0.5);


figure;
plot(Y_vals, f_Y, 'm', 'LineWidth', 2);
xlabel('Y'); ylabel('p(Y)');
title('Estimated Marginal Density p(Y)');
grid on;
%% Step 4: Compute Conditional Density p(X | Y)
y1 = 3; % Given Y = y1
tolerance = 0.2; % Small range around y1
X_given_Y = X(abs(Y - y1) < tolerance); % Select X values near y1

% Estimate p(X | Y = y1) using KDE
[f_X_given_Y, X_vals] = ksdensity(X_given_Y, 'Bandwidth', 0.5, 'NumPoints', 10000);

% Plot p(X | Y = y1)
figure;
subplot(2,1,1);
plot(X_vals, f_X_given_Y, 'b', 'LineWidth', 2);
xlabel('X'); ylabel('p(X | Y)');
title(['Conditional Density p(X | Y = ', num2str(y1), ')']);
grid on;

%% Step 5: Compute CDF of p(X | Y)
CDF_X_given_Y = cumsum(f_X_given_Y) / sum(f_X_given_Y);

% Plot CDF of p(X | Y)
subplot(2,1,2);
plot(X_vals, CDF_X_given_Y, 'r', 'LineWidth', 2);
xlabel('X'); ylabel('CDF');
title(['Cumulative Distribution Function CDF(X | Y = ', num2str(y1), ')']);
grid on;
%% Step 6: Sample from p(X | Y) using Inverse Transform Sampling
u = rand(100000,1); % Generate uniform random number
X_sampled = interp1(CDF_X_given_Y, X_vals, u, 'linear', 'extrap');

figure;
[f_X_sampled, X_sampled_vals] = ksdensity(X_sampled, 'Bandwidth', 0.5); % Estimate density of samples
plot(X_sampled_vals, f_X_sampled, 'g', 'LineWidth', 2);
xlabel('X'); ylabel('p(X | Y)');
title('Estimated PDF of Sampled X Given Y = y_1');
grid on;
