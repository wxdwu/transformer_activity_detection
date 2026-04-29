function [eta, etaPrimeAvg] = threshPrimeThreshComplexGaussian_Genie(y,N,M,sigma,lambda_vec,pLS)
    eta = zeros(N,M);
    etaPrimeAvg = zeros(M,M);
    
    sigma2 = sigma^2;
    
    for n=1:N
        %以此处为核心修改点：获取当前用户的 lambda
        lam_n = lambda_vec(n);
        p_n = pLS(n); % 假设 pLS 也是 N x 1 的向量
        
        % 1. 维纳滤波系数 (活跃时的幅度估计系数)
        a = p_n / (p_n + sigma2); 
        
        % 2. 计算似然比相关的项 (对应后验概率)
        % 原公式: b = (1-lambda)/lambda * ((p+sigma^2)/sigma^2)^M
        term_power = ((p_n + sigma2) / sigma2)^M;
        b = (1 - lam_n) / lam_n * term_power;
        
        % c 是指数项系数
        c = p_n / (sigma2 * (p_n + sigma2));
        
        % 3. 计算中间变量 t
        % y(n,:) * y(n,:)' 计算的是行向量的模平方
        y_norm2 = sum(abs(y(n,:)).^2); 
        
        t0 = b * exp(-c * y_norm2);
        t = 1 + t0;
        
        % 4. 计算 MMSE 估计值 eta (Expectation)
        % coeff1 实际上等于: (后验活跃概率) * (维纳系数 a)
        % 后验活跃概率 pi_post = 1 / (1 + t0) = 1/t
        coeff1 = a / t; 
        
        eta(n,:) = coeff1 * y(n,:);    
        
        % 5. 计算导数项 (用于 Onsager 校正)
        coeff0 = a^2 / sigma2;
        % 这是一个标量导数近似 (针对 MIMO 情况的平均)
        % etaPrimeMtx = coeff1*eye(M) + coeff0*(y(n,:)'*y(n,:))*t0/t^2;
        % 为了保持原代码逻辑，这里累加矩阵
        
        % 优化：避免循环内的矩阵乘法，直接算对角线均值贡献
        % 原代码逻辑比较耗时，但为了完全复现逻辑保持不变：
        term_derivative = coeff0 * (y(n,:)' * y(n,:)) * (t0 / t^2);
        etaPrimeAvg = etaPrimeAvg + (coeff1 * eye(M) + term_derivative);
    end
    etaPrimeAvg = etaPrimeAvg/N;
end