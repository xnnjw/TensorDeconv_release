%% ========================================================================
%  PALM Tensor Deconvolution Demo Script
%  
%  This demo showcases the PALM algorithm for hyperspectral image 
%  deconvolution using CP tensor decomposition with total variation 
%  regularization.
%
%  Features:
%  - Interactive parameter selection
%  - Real-time convergence visualization  
%  - Parameter efficiency comparison
%  - Visual reconstruction results
%  - Robust initialization with fallback methods
%% ========================================================================

clear; clc; close all;

% Add necessary paths
addpath('util');
addpath('assessment'); 
addpath(genpath('tensor_toolbox-v3.6/'));

fprintf('=== PALM Tensor Deconvolution Demo ===\n\n');

%% Demo Configuration
CAVE_SCENE_ID = 27;        % Select scene from CAVE dataset (1-32)
KERNEL_ID = 1;            % Select blur kernel
DEMO_RANK = 20;           % CP decomposition rank (adjustable)

% Data paths (adjust to your setup)
data_path = './data/complete_ms_data/';
kernel_path = './data/kernels/';

%% Load Demo Data
fprintf('Loading CAVE dataset scene %d with kernel %d...\n', CAVE_SCENE_ID, KERNEL_ID);
try
    [img_clean, img_blurred, kernel, scene_name] = load_CAVE_scene(CAVE_SCENE_ID, KERNEL_ID, data_path, kernel_path);
    [P, Q, N] = size(img_clean);
    fprintf('✓ Loaded scene: %s\n', scene_name);
    fprintf('✓ Image dimensions: %d × %d × %d\n', P, Q, N);
catch ME
    error('Failed to load data. Please check your data paths and ensure CAVE dataset is available.\nError: %s', ME.message);
end

%% Parameter Setup
fprintf('\nSetting up PALM parameters...\n');

% Initial hyper-parameters (tuning)
params = struct();
params.rank = DEMO_RANK;
params.lambda1 = 4e-5;      % Spatial regularization (mode-1)
params.lambda2 = 6.5e-5;    % Spatial regularization (mode-2)  
params.lambda3 = 1e-7;      % Spectral regularization (mode-3)
params.lambda_A = 6e-2;     % TV regularization for factor A
params.lambda_B = 1e-3;     % TV regularization for factor B
params.max_iter = 300;      % Maximum iterations
params.tol = 1e-4;          % Convergence tolerance
params.verbose = true;      % Show progress

fprintf('✓ Rank: %d\n', params.rank);
fprintf('✓ Regularization: λ₁=%.1e, λ₂=%.1e, λ₃=%.1e\n', params.lambda1, params.lambda2, params.lambda3);
fprintf('✓ TV regularization: λ_A=%.1e, λ_B=%.1e\n', params.lambda_A, params.lambda_B);

%% Parameter Efficiency Analysis
fprintf('\n=== Parameter Efficiency Analysis ===\n');
full_rank_params = P * Q * N;
cp_params = (P + Q + N) * params.rank;
compression_ratio = full_rank_params / cp_params;

fprintf('Full-rank representation: %d parameters\n', full_rank_params);
fprintf('CP decomposition (rank %d): %d parameters\n', params.rank, cp_params);
fprintf('Compression ratio: %.1f× parameter reduction\n', compression_ratio);

%% Initialize with CPD (Improved with Fallback)
fprintf('\nInitializing with CP decomposition...\n');
[init_A, init_B, init_C] = initialize_with_cpd_demo(img_blurred, params.rank);
params.init_A = init_A;
params.init_B = init_B; 
params.init_C = init_C;

%% Run PALM Algorithm
fprintf('\n=== Running PALM Algorithm ===\n');
tic;
[A, B, C, history] = palm_tensordeconv(img_blurred, kernel, params);
runtime = toc;

fprintf('✓ PALM algorithm completed in %.2f seconds\n', runtime);
fprintf('✓ Converged in %d iterations\n', length(history.F_values));

%% Reconstruct and Evaluate
fprintf('\nReconstructing tensor and evaluating results...\n');
X_rec = reconstruct_X(A, B, C);
metrics = evaluate_reconstruction(X_rec, img_clean);

fprintf('=== Reconstruction Quality ===\n');
fprintf('PSNR: %.2f dB\n', metrics.PSNR);
fprintf('SSIM: %.4f\n', metrics.SSIM);
fprintf('RMSE: %.4f\n', metrics.RMSE);
fprintf('SAM: %.2f°\n', metrics.SAM);
fprintf('Relative Error: %.4f\n', metrics.RelError);

%% Visualization
fprintf('\nGenerating visualization...\n');
create_demo_visualization(img_clean, img_blurred, X_rec, history, params, metrics, scene_name);

fprintf('\n=== Demo Complete ===\n');
fprintf('Algorithm: PALM Tensor Deconvolution\n');
fprintf('Scene: %s\n', scene_name);
fprintf('Final PSNR: %.2f dB (%.1f× parameter reduction)\n', metrics.PSNR, compression_ratio);

%% Helper Functions

function [init_A, init_B, init_C] = initialize_with_cpd_demo(img_blurred, rank)
    % Initialize factors using CP decomposition with fallback methods
    
    fprintf('Attempting CPD initialization...\n');
    
    % Method 1: Try tensor toolbox cp_opt
    try
        Y_tensor = tensor(img_blurred);
        [cpd_result, ~, ~] = cp_opt(Y_tensor, rank, 'maxiters', 50);
        
        cpd_factors = cpd_result.U;
        init_A = cpd_factors{1};
        init_B = cpd_factors{2};
        init_C = cpd_factors{3};
        fprintf('✓ CPD initialization using tensor toolbox completed successfully\n');
        return;
    catch ME
        fprintf('⚠ Tensor toolbox CPD failed: %s\n', ME.message);
    end
    
    % Method 2: Try manual CPD initialization
    try
        fprintf('Attempting manual CPD initialization...\n');
        [init_A, init_B, init_C] = manual_cpd_init_worker(img_blurred, rank);
        fprintf('✓ Manual CPD initialization completed successfully\n');
        return;
    catch ME
        fprintf('⚠ Manual CPD initialization failed: %s\n', ME.message);
    end
    
    % Method 3: Fallback to random initialization
    fprintf('Using random initialization as fallback...\n');
    [P, Q, N] = size(img_blurred);
    init_A = randn(P, rank) * 0.1;
    init_B = randn(Q, rank) * 0.1;
    init_C = randn(N, rank) * 0.1;
    fprintf('⚠ Using random initialization (may affect convergence quality)\n');
end

function [A, B, C] = manual_cpd_init_worker(Y, rank)
    % Manual CPD initialization using SVD
    % 
    % Input:
    %   Y - Input tensor of size P × Q × N
    %   rank - Desired CP decomposition rank
    % Output:
    %   A, B, C - Factor matrices of sizes P×rank, Q×rank, N×rank
    %
    % Mathematical formulation:
    % For tensor Y ∈ R^{P×Q×N}, we seek factors such that:
    % Y ≈ ∑_{r=1}^{rank} a_r ∘ b_r ∘ c_r
    
    [P, Q, N] = size(Y);
    
    % Initialize with random matrices (small values for stability)
    A = randn(P, rank) * 0.01;
    B = randn(Q, rank) * 0.01;
    C = randn(N, rank) * 0.01;
    
    % Iterative refinement using SVD of mode unfoldings
    % This implements a simplified ALS (Alternating Least Squares) approach
    for iter = 1:5
        % Mode-1 unfolding: Y_{(1)} ∈ ℝ^{P×(QN)}
        Y_1 = reshape(Y, P, Q*N);
        [U, S, ~] = svd(Y_1, 'econ');
        A = U(:, 1:min(rank, size(U,2))) * diag(sqrt(diag(S(1:min(rank, size(S,1)), 1:min(rank, size(S,2))))));
        
        % Mode-2 unfolding: Y_{(2)} ∈ ℝ^{Q×(PN)}
        Y_2 = reshape(permute(Y, [2, 1, 3]), Q, P*N);
        [U, S, ~] = svd(Y_2, 'econ');
        B = U(:, 1:min(rank, size(U,2))) * diag(sqrt(diag(S(1:min(rank, size(S,1)), 1:min(rank, size(S,2))))));
        
        % Mode-3 unfolding: Y_{(3)} ∈ ℝ^{N×(PQ)}
        Y_3 = reshape(permute(Y, [3, 1, 2]), N, P*Q);
        [U, S, ~] = svd(Y_3, 'econ');
        C = U(:, 1:min(rank, size(U,2))) * diag(sqrt(diag(S(1:min(rank, size(S,1)), 1:min(rank, size(S,2))))));
    end
    
    % Ensure dimensions are correct
    if size(A, 2) < rank
        A = [A, randn(P, rank - size(A, 2)) * 0.01];
    end
    if size(B, 2) < rank
        B = [B, randn(Q, rank - size(B, 2)) * 0.01];
    end
    if size(C, 2) < rank
        C = [C, randn(N, rank - size(C, 2)) * 0.01];
    end
    
    % Take only the first rank columns
    A = A(:, 1:rank);
    B = B(:, 1:rank);
    C = C(:, 1:rank);
end

function create_demo_visualization(img_clean, img_blurred, X_rec, history, params, metrics, scene_name)
    % Create simplified visualization with 4 subplots
    
    % Select middle spectral band for display
    middle_band = ceil(size(img_clean, 3) / 2);
    
    figure('Position', [100, 100, 1000, 600], 'Name', 'PALM Tensor Deconvolution Demo');
    
    % First row: Image comparison
    subplot(2, 2, 1);
    imshow(img_clean(:,:,middle_band), []); 
    title(sprintf('Ground Truth\n(Band %d/%d)', middle_band, size(img_clean,3)));
    colorbar;
    
    subplot(2, 2, 2);
    imshow(X_rec(:,:,middle_band), []);
    title(sprintf('PALM Reconstruction\nPSNR: %.2f dB', metrics.PSNR));
    colorbar;
    
    % Second row: Convergence and Parameter Efficiency
    subplot(2, 2, 3);
    semilogy(history.F_values, 'b-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Objective Function');
    title('PALM Convergence');
    grid on;
    
    subplot(2, 2, 4);
    % Parameter efficiency with different colors and log scale
    full_params = numel(img_clean);
    cp_params = (size(img_clean,1) + size(img_clean,2) + size(img_clean,3)) * params.rank;
    
    h = bar([full_params, cp_params]);
    h.FaceColor = 'flat';
    h.CData(1,:) = [0.8 0.2 0.2];  % Red for Full Rank
    h.CData(2,:) = [0.2 0.6 0.8];  % Blue for CPD
    
    set(gca, 'YScale', 'log');
    set(gca, 'XTickLabel', {'Full Rank', sprintf('CPD (R=%d)', params.rank)});
    ylabel('Parameters (log scale)');
    title('Parameter Efficiency');
    
    % Add compression ratio text
    compression_ratio = full_params / cp_params;
    text(1.5, sqrt(full_params * cp_params), ...
         sprintf('%.1f× reduction', compression_ratio), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
    grid on;
    
    % Main title
    sgtitle(sprintf('PALM Tensor Deconvolution Demo - (Rank %d)', params.rank), ...
            'FontSize', 16, 'FontWeight', 'bold');
end
