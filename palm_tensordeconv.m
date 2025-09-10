%% ========================================================================
%  PALM for Tensor Deconvolution - Modified Version with Separate Lambda
%  
%  Main function:
%  [A, B, C, history] = palm_tensordeconv(Y, H, options)
%
%  Author: Modified from Xinjue Wang's version
%  Date: 2025/08/26
%% ========================================================================

function [A, B, C, history] = palm_tensordeconv(Y, H, options)
    % PALM algorithm for tensor deconvolution with separate lambda parameters
    %
    % Inputs:
    %   Y - Observed blurred tensor (P×Q×N)  
    %   H - Blur kernels (5×5×N or single kernel)
    %   options - struct with fields:
    %       .rank (R) - CP decomposition rank
    %       .lambda1, .lambda2, .lambda3 - Frobenius regularization for A, B, C
    %       .lambda_A, .lambda_B - TV regularization parameters
    %       .max_iter, .tol - convergence settings
    %       .init_A, .init_B, .init_C - (optional) initial factors
    %
    % Outputs:
    %   A, B, C - Factor matrices
    %   history - convergence history
    
    %% --- 1. Parameters and Preprocessing ---
    [P, Q, N] = size(Y);
    R = options.rank;
    lambda1 = options.lambda1;  % A Frobenius regularization
    lambda2 = options.lambda2;  % B Frobenius regularization
    lambda3 = options.lambda3;  % C Frobenius regularization
    lambda_A = options.lambda_A; % A TV regularization
    lambda_B = options.lambda_B; % B TV regularization
    max_iter = options.max_iter;
    tol = options.tol;
    beta = 0.4;  % backtracking parameter
    eta = 0.8;   % step size increase factor
    
    % Verbose setting (default: true)
    % if ~isfield(options, 'verbose')
    options.verbose = true;
    % end
    
    % Compute FFT of observations and kernels
    Y_hat = fft2(Y);
    H_hat = compute_H_hat(H, P, Q, N);
    
    %% --- 2. Initialization ---
    if isfield(options, 'init_A') && ~isempty(options.init_A)
        A = options.init_A;
    else
        A = rand(P, R);
    end
    
    if isfield(options, 'init_B') && ~isempty(options.init_B)
        B = options.init_B;
    else
        B = rand(Q, R);
    end
    
    if isfield(options, 'init_C') && ~isempty(options.init_C)
        C = options.init_C;
    else
        C = rand(N, R);
    end
    
    % Initial step sizes
    c = 0.1; d = 0.1; e = 0.1;
    
    % History trackers
    history.F_values = [];
    history.errors = [];
    
    %% --- 3. PALM Main Iteration ---
    if options.verbose
        fprintf('Starting PALM iterations...\n');
    end
    
    for k = 1:max_iter
        A_prev = A; B_prev = B; C_prev = C;
        
        % --- Update Block A ---
        G_A = compute_grad_A(A, B, C, H_hat, Y_hat, lambda1);
        f_A = @(A_in) compute_f(A_in, B, C, H_hat, Y_hat, lambda1, lambda2, lambda3);
        prox_A_handle = @(U, t) prox_A(U, t, lambda_A);
        [A, c] = backtrack_ls(A, G_A, c/eta, beta, f_A, prox_A_handle);
        
        % --- Update Block B ---
        G_B = compute_grad_B(A, B, C, H_hat, Y_hat, lambda2);
        f_B = @(B_in) compute_f(A, B_in, C, H_hat, Y_hat, lambda1, lambda2, lambda3);
        prox_B_handle = @(U, t) prox_B(U, t, lambda_B);
        [B, d] = backtrack_ls(B, G_B, d/eta, beta, f_B, prox_B_handle);
        
        % --- Update Block C ---
        G_C = compute_grad_C(A, B, C, H_hat, Y_hat, lambda3);
        f_C = @(C_in) compute_f(A, B, C_in, H_hat, Y_hat, lambda1, lambda2, lambda3);
        [C, e] = backtrack_ls(C, G_C, e/eta, beta, f_C, @prox_C);
        
        % --- Record metrics ---
        F_k = compute_F(A, B, C, H_hat, Y_hat, lambda1, lambda2, lambda3, lambda_A, lambda_B);
        history.F_values = [history.F_values; F_k];
        
        % --- Check convergence ---
        diff = norm(A - A_prev, 'fro') / norm(A_prev, 'fro') + ...
               norm(B - B_prev, 'fro') / norm(B_prev, 'fro') + ...
               norm(C - C_prev, 'fro') / norm(C_prev, 'fro');
        
        if options.verbose && mod(k, 5) == 0
            fprintf('Iter %d: F = %.4e, Rel. Change = %.3e\n', k, F_k, diff);
        end
        
        if diff < tol
            if options.verbose
                fprintf('Convergence reached at iteration %d.\n', k);
            end
            break;
        end
    end
end

%% ========================================================================
%  Helper functions
%% ========================================================================

function H_hat = compute_H_hat(H, P, Q, N)
    % FFT
    H_hat = zeros(P, Q, N);
    
    if size(H, 3) == N
        % H is 3D array with different kernel for each channel
        for i = 1:N
            H_hat(:,:,i) = psf2otf(H(:,:,i), [P, Q]);
        end
    else
        % Single kernel for all channels
        for i = 1:N
            H_hat(:,:,i) = psf2otf(H, [P, Q]);
        end
    end
end

function F_val = compute_F(A, B, C, H_hat, Y_hat, lambda1, lambda2, lambda3, lambda_A, lambda_B)
    % F = f + g 
    f_val = compute_f(A, B, C, H_hat, Y_hat, lambda1, lambda2, lambda3);
    g_A_val = lambda_A * sum(arrayfun(@(r) tv_norm(A(:,r)), 1:size(A,2)));
    g_B_val = lambda_B * sum(arrayfun(@(r) tv_norm(B(:,r)), 1:size(B,2)));
    F_val = f_val + g_A_val + g_B_val;
end

function f_val = compute_f(A, B, C, H_hat, Y_hat, lambda1, lambda2, lambda3)
    % Smooth part
    X = reconstruct_X(A, B, C);
    X_hat = fft2(X);
    
    % FFT Residual
    R_hat = H_hat .* X_hat - Y_hat;
    f_conv = 0.5 * sum(abs(R_hat).^2, 'all');
    
    % Frobenius
    f_reg = lambda1 * sum(A.^2, 'all') + lambda2 * sum(B.^2, 'all') + lambda3 * sum(C.^2, 'all');
    
    f_val = f_conv + f_reg;
end

function X = reconstruct_X(A, B, C)
    % Pagemtimes 
    [P, R] = size(A); [Q, ~] = size(B); [N, ~] = size(C);
    
    A_pages = repmat(A, [1, 1, N]);     % P×R×N
    B_pages = repmat(B', [1, 1, N]);    % R×Q×N
    
    C_diag_pages = zeros(R, R, N);
    for i = 1:N
        C_diag_pages(:,:,i) = diag(C(i,:));
    end
    
    temp = pagemtimes(A_pages, C_diag_pages);
    X = pagemtimes(temp, B_pages);
end

function tv = tv_norm(x)
    % 1D Total Variation norm
    tv = sum(abs(diff(x)));
end

function G_A = compute_grad_A(A, B, C, H_hat, Y_hat, lambda1)
    % Gradient of A
    [P, Q, N] = size(Y_hat); R = size(A, 2);
    
    X = reconstruct_X(A, B, C);
    X_hat = fft2(X);
    
    R_hat = H_hat .* X_hat - Y_hat;
    conj_H_R = conj(H_hat) .* R_hat;
    S = real(ifft2(conj_H_R));
    
    B_pages = repmat(B, [1, 1, N]);
    SB = pagemtimes(S, B_pages);
    
    G_A_conv = zeros(P, R);
    for i = 1:N
        G_A_conv = G_A_conv + SB(:,:,i) .* C(i,:);
    end
    
    G_A = G_A_conv + 2 * lambda1 * A;  
end

function G_B = compute_grad_B(A, B, C, H_hat, Y_hat, lambda2)
    % Gradient of B
    [P, Q, N] = size(Y_hat); R = size(B, 2);
    
    X = reconstruct_X(A, B, C);
    X_hat = fft2(X);
    
    R_hat = H_hat .* X_hat - Y_hat;
    conj_H_R = conj(H_hat) .* R_hat;
    S = real(ifft2(conj_H_R));
    
    S_T = permute(S, [2, 1, 3]);
    A_pages = repmat(A, [1, 1, N]);
    STA = pagemtimes(S_T, A_pages);
    
    G_B_conv = zeros(Q, R);
    for i = 1:N
        G_B_conv = G_B_conv + STA(:,:,i) .* C(i,:);
    end
    
    G_B = G_B_conv + 2 * lambda2 * B;  %lambda2
end

function G_C = compute_grad_C(A, B, C, H_hat, Y_hat, lambda3)
    % Gradient of C
    [P, Q, N] = size(Y_hat); R = size(C, 2);
    
    X = reconstruct_X(A, B, C);
    X_hat = fft2(X);
    
    R_hat = H_hat .* X_hat - Y_hat;
    conj_H_R = conj(H_hat) .* R_hat;
    S = real(ifft2(conj_H_R));
    
    A_T_pages = repmat(A', [1, 1, N]);
    B_pages = repmat(B, [1, 1, N]);
    
    AS = pagemtimes(A_T_pages, S);
    ASB = pagemtimes(AS, B_pages);
    
    G_C_conv = zeros(N, R);
    for i = 1:N
        G_C_conv(i,:) = diag(ASB(:,:,i))';
    end
    
    G_C = G_C_conv + 2 * lambda3 * C;  % lambda3
end

function [Z_new, t_new] = backtrack_ls(Z, G_Z, t_init, beta, compute_f_handle, prox_g_handle)
    % Backtraking for stepsize
    t = t_init;
    f_current = compute_f_handle(Z);
    
    while true
        U = prox_g_handle(Z - t * G_Z, t);
        f_LHS = compute_f_handle(U);
        f_RHS = f_current + sum((U - Z) .* G_Z, 'all') + (1/(2*t)) * norm(U - Z, 'fro')^2;
        
        if f_LHS <= f_RHS
            break;
        end
        t = beta * t;
    end
    
    Z_new = U;
    t_new = t;
end

function A_new = prox_A(U, t, lambda_A)
    % Proximal op of A
    A_new = U;
    
    % TV_Condat_v2
    if exist('TV_Condat_v2', 'file')
        for i = 1:size(U,2)
            A_new(:,i) = TV_Condat_v2(U(:,i), t * lambda_A);
        end
    else
        % 
        A_new = max(U, 0);
    end
end

function B_new = prox_B(U, t, lambda_B)
    % Proximal op of B
    B_new = U;
    
    if exist('TV_Condat_v2', 'file')
        for i = 1:size(U,2)
            B_new(:,i) = TV_Condat_v2(U(:,i), t * lambda_B);
        end
    else
        B_new = max(U, 0);
    end
end

function C_new = prox_C(U, ~)
    % Project C to nonnegative orthant
    C_new = max(U, 0);
end