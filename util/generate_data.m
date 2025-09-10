function [A, B, C, Y, H_hat, Y_hat] = generate_data(P, Q, N, R)
    % Generate synthetic problem data
    A = max(0.1, rand(P, R)); % Non-negative random matrices
    B = max(0.1, rand(Q, R));
    C = max(0.1, rand(N, R));

    X = reconstruct_X(A, B, C);
    H_kernels = randn(5, 5, N); % Random convolution kernels
    
    Y = zeros(P, Q, N);
    for i = 1:N
        Y(:,:,i) = conv2(X(:,:,i), H_kernels(:,:,i), 'same') + 0.01 * randn(P, Q); % Convolve and add noise
    end
    
    H_hat = zeros(P, Q, N);
    for i = 1:N
        H_hat(:,:,i) = psf2otf(H_kernels(:,:,i), [P, Q]); % Use psf2otf for correct FFT of kernel
    end
    Y_hat = fft2(Y);
end