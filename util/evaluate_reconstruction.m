function metrics = evaluate_reconstruction(X_rec, X_true)
    % 计算重建质量的各种指标
    [psnr, rmse, sam, ssim, ergas] = quality_assessment(single(im2uint8(X_true)), single(im2uint8(X_rec)), 0);
    metrics.PSNR = psnr;
    metrics.SSIM = ssim;
    metrics.RMSE = rmse;
    metrics.SAM = sam;
    
    % 相对误差
    metrics.RelError = norm(X_rec(:) - X_true(:)) / norm(X_true(:));
end