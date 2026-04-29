%% Genie-Aided CAMP 验证脚本：Monte Carlo 100 次平均
clear; clc;

% --- 1. 参数设置 ---
N = 200;        % 用户数
L = 30;         % 导频长度
M = 32;         % 天线数
sparsity = 0.1; % 稀疏度，即活跃用户比例
SNR = 20;
max_iter = 40;
MC_times = 1; % Monte Carlo 实验次数

% 大尺度衰落和发射功率参数
cell_radius_m = 250.0;
pmax_dbm = 23.0;

% 用于累计每次 Monte Carlo 的 NMSE 曲线
nmse_blind_all = zeros(max_iter, MC_times);
nmse_genie_all = zeros(max_iter, MC_times);

% --- 2. Monte Carlo 循环 ---
for mc = 1:MC_times
    fprintf('Monte Carlo 实验 %d / %d\n', mc, MC_times);

    % 生成活跃用户指示器 (0/1)
    active_idx = zeros(N, 1);
    k = round(N * sparsity);
    perm = randperm(N);
    active_idx(perm(1:k)) = 1;

    % 信号功率 (Fading / Beta)
    % 用户在圆形小区内均匀分布：r = R * sqrt(u)
    distances = cell_radius_m * sqrt(max(rand(N, 1), 1e-8));
    d_km = max(distances / 1000.0, 1e-3);
    pl_db = 128.1 + 37.6 * log10(d_km);
    beta = 10.^(-pl_db / 10.0);
    beta_min = min(beta);
    pmax_w = 10^((pmax_dbm - 30.0) / 10.0);
    p = pmax_w * (beta_min ./ max(beta, 1e-16));
    pg = p .* beta;

    % 生成真实信号 X
    X = zeros(N, M);
    active_users = find(active_idx);
    H = (randn(k, M) + 1i*randn(k, M)) / sqrt(2);
    X(active_users, :) = sqrt(pg(active_users)) .* H;

    % 感知矩阵 A (Phi)
    A = (randn(L, N) + 1i*randn(L, N)) / sqrt(2 * L);

    % 接收信号 Y
    noise_power = 10^(-SNR/10) * (sum(pg(active_users)) / L);

    sigma_w = sqrt(noise_power);
    Noise = (randn(L, M) + 1i*randn(L, M)) / sqrt(2) * sigma_w;
    Y = A * X + Noise;

    % --- 3. 运行算法对比 ---

    % 场景 A: 普通 CAMP (Blind)
    lambda_blind = sparsity * ones(N, 1);
    [~, ~, mse_blind, ~, ~] = CAMP_Genie(A, Y, X, max_iter, lambda_blind, pg, sigma_w);
    nmse_blind_all(:, mc) = 10 * log10(mse_blind(:) ./ (norm(X, 'fro')^2 / N / M));

    % 场景 B: Genie-Aided CAMP (上帝视角)
    epsilon = 1e-9;
    lambda_genie = active_idx;
    lambda_genie(lambda_genie == 0) = epsilon;
    lambda_genie(lambda_genie == 1) = 1 - epsilon;

    [~, ~, mse_genie, ~, ~] = CAMP_Genie(A, Y, X, max_iter, lambda_genie, pg, sigma_w);
    nmse_genie_all(:, mc) = 10 * log10(mse_genie(:) ./ (norm(X, 'fro')^2 / N / M));
end

% --- 4. 计算 100 次 Monte Carlo 平均值 ---
nmse_blind_avg = mean(nmse_blind_all, 2);
nmse_genie_avg = mean(nmse_genie_all, 2);

% 如需在命令行查看平均结果，可取消下面两行注释
% disp('Standard CAMP 平均 NMSE (dB):'); disp(nmse_blind_avg.');
% disp('Genie-Aided CAMP 平均 NMSE (dB):'); disp(nmse_genie_avg.');

% --- 5. 绘图 ---
figure;
plot(nmse_blind_avg, '-ob', 'LineWidth', 1.5); hold on;
plot(nmse_genie_avg, '-xr', 'LineWidth', 2);
grid on;
legend('Standard CAMP (Prior=0.1)', 'Genie-Aided CAMP (True Prior)');
xlabel('Iteration');
ylabel('Average NMSE (dB)');
title(sprintf('Genie-Aided Analysis, Monte Carlo Average over %d Runs', MC_times));
