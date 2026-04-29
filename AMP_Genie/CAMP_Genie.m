function [xnoise,x,mse,tau_real,tau_est] = CAMP_Genie(A,y,xsig,maxN_itera,lambda_vec,fading,sigma_w)
% CAMP_Genie: 支持输入每个用户独立先验概率的 CAMP 算法
% 输入:
%   lambda_vec: N x 1 向量，每个用户的活跃概率 (0~1)

N = size(A,2);
M = size(y,2);
L = size(y,1);

% --- 数值稳定性保护 ---
% 防止 lambda 为 0 或 1 导致 log 或除法错误
epsilon = 1e-9;
lambda_vec(lambda_vec < epsilon) = epsilon;
lambda_vec(lambda_vec > 1-epsilon) = 1-epsilon;

z = y;
x = zeros(N,M);

% 初始化残差能量
sum_gain = sum(sum(abs(y).^2));
tau = sqrt(sum_gain)*sqrt(1/(M*L));  

mse = zeros(maxN_itera,1);
tau_real = zeros(maxN_itera,1);
tau_est = zeros(maxN_itera,1);
tau_real(1) = tau;
tau_est(1) = tau;

xold = zeros(N,M); % 初始化 xold

for i = 1:maxN_itera 
    % 1. 线性步：计算输入到去噪器的值 (r = x + A'z)
    input = A'*z + x;
    
    % 2. 非线性步：去噪 (传入向量 lambda_vec)
    [x_new, avgxprime] = threshPrimeThreshComplexGaussian_Genie(input,N,M,tau,lambda_vec,fading);
    
    % 3. 阻尼 (Damping) - 保持与原代码一致
    if i==1
        x = x_new;
    else
        x = 0.95*x_new + 0.05*xold;
    end
    xold = x;
    
    % 4. 计算 MSE (用于性能监控)
    % 优化计算速度，避免循环
    mse(i) = sum(sum(abs(x - xsig).^2)) / (N*M);
    
    % 5. 更新状态演进 (State Evolution) 的 tau
    % 注意：这里使用 mse(i) 作为 x 的误差估计是假设已知 xsig 的
    % 实际算法中通常使用 z 的能量来估计，但为了保持原代码逻辑不变：
    tau_real(i+1) = sqrt(sigma_w^2 + N/L*mse(i));
    
    % 6. 更新残差 z (Onsager Correction)
    z = y - A*x + N/L*z*avgxprime; 
    
    % 7. 重新估计噪声水平 (自适应)
    sum_gain = sum(sum(abs(z).^2));
    tau = sqrt(sum_gain)*sqrt(1/(M*L));  
    
    if i ~= maxN_itera
        tau_est(i+1) = tau;
    end
end
xnoise = A'*z + x;
end