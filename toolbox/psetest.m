function psetest(m, n, k, r)
    X = randn(m, n, k);
    X_s = Tensor2SmallCircM(X);
    iterationNums = 5;
    minSamplingRate = 20;
    nums = floor(m*k*n*minSamplingRate/100);
    A_all = randn( nums, m*k*n);
    A_all_t = randn(nums, m*k*n);
    y = A_all*X_s(:);
    t1 = clock;
    testTS(X, X_s, A_all, A_all_t, y, iterationNums, r);    % 10为迭代次数，r为rank
    t2 = clock;
    t = etime(t2, t1);
    fprintf("%d, %d, %d, %d, time %f\n", m, n, k, r, t);
end
