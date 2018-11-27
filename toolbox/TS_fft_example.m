function TS_fft_example()
    %% 构造人工数据
    L = randn(40, 1, 6);
    R = randn(1, 40, 6);
    X = tProd(L, R, true);
    [~, r ,~] = size(L);
    % load('X_60_60_15_decorate.mat');
    % r = 1;
    X = fft(X);
    [m, n, k] = size(X);
    sz_X = [m, n, k];
    X_s = Tensor2SmallCircM(X);
    minSamplingRate = 20;
    minSamplingRate_copy = minSamplingRate;
    maxSamplingRate = 80;
    samplingNums = (maxSamplingRate - minSamplingRate)/5 + 1;
    error = zeros(samplingNums, 1);
    errot_index = 1;
    iterationNums = 5;
    error_all = zeros(iterationNums, samplingNums);
    while(minSamplingRate <= maxSamplingRate)  
        nums = floor(m*k*n*minSamplingRate/100);
        A_all = sparse(m*k*n, nums);       %所有的 A_s_v 向量叠在一起组成一个大矩阵
        %% 构造满足衰减特性的采样tensor
        for i = 1 : nums
            sampling_tensor = generate_sampling_tensor(sz_X, 20);
            A_s = Tensor2SmallCircM(sampling_tensor);
            A_s = sparse(A_s);
            A_s_t = A_s.';

            A_s_v = A_s(:);                    %  A_s_v 表示A_s的向量化
            A_s_t_v = A_s_t(:);

            A_all(:, i) = A_s_v;
            A_all_t(:, i) = A_s_t_v;
        end
        A_all = A_all.';
        y = sparse(A_all*X_s(:));
        X_re = A_all \ y;
        rse = norm(X_s(:) - X_re(:)) / norm(X_s(:));
        fprintf('based error is %e\n',rse);
        fprintf('based y error is %e\n', norm(A_all * X_re - y(:)) / norm(y(:)));
        

        [RSE,error_single] = TS_fft(X, A_all, y, iterationNums, r, minSamplingRate);    % 10为迭代次数，r为rank
        error_all(:, errot_index) = error_single;
        error(errot_index, 1) = RSE;
        minSamplingRate = minSamplingRate + 5;
        errot_index = errot_index  + 1;
    end
    %% 画误差曲线                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    %plotRSE(DCTorFFt, minSamplingRate_copy, maxSamplingRate, error');
    %plot_shou_lian_lv(DCTorFFt, iterationNums, error_all(:,13)');
end





