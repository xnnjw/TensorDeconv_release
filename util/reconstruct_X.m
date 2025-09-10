function X = reconstruct_X(A, B, C)
    % pagemtimes 方法
    [P, R] = size(A); [Q, ~] = size(B); [N, ~] = size(C);
    % 准备页矩阵
    A_pages = repmat(A, [1, 1, N]);     % P×R×N
    B_pages = repmat(B', [1, 1, N]);    % R×Q×N
    % 创建对角矩阵页
    C_diag_pages = zeros(R, R, N);
    for i = 1:N
        C_diag_pages(:,:,i) = diag(C(i,:));
    end
    % 页矩阵乘法
    temp = pagemtimes(A_pages, C_diag_pages);
    X = pagemtimes(temp, B_pages);
end