function TStest(m, n, k, r)
    %% 构造人工数据
    L = randn(m, r, k);
    R = randn(r, n, k);
    X = tProd(L, R, true);
    X = fft(X);
    % load('X_60_60_15_decorate.mat');
    % r = 1;
    sz_X = [m, n, k];
    X_s = Tensor2SmallCircM(X);
    minSamplingRate = 20;
    minSamplingRate_copy = minSamplingRate;
    maxSamplingRate = 80;
    samplingNums = (maxSamplingRate - minSamplingRate)/5 + 1;
    error = zeros(samplingNums, 1);
    errot_index = 1;
    iterationNums = 4;
    error_all = zeros(iterationNums, samplingNums);
    while(minSamplingRate <= maxSamplingRate)  
       nums = floor(m*k*n*minSamplingRate/100);
        A_all = sparse(m*k*n, nums);       %所有的 A_s_v 向量叠在一起组成一个大矩阵
        A_all_t = sparse(m*k*n, nums);
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
        A_all_t = A_all_t.';
        y = sparse(A_all*X_s(:));

        t1 = clock;
        testTS(X, X_s, A_all, A_all_t, y, iterationNums, r);    % 10为迭代次数，r为rank
        t2 = clock;
        t = etime(t2, t1);
        fprintf('size[%d,%d, %d, %d] takes time %f\n', m, n, k, r, t);
    end
    %% 画误差曲线                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
end





