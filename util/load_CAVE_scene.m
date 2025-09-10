function [img_clean, img_blurred, kernel, scene_name] = load_CAVE_scene(scene_idx, kernel_idx, data_path, kernel_path)
    % 加载CAVE数据集的特定场景和模糊核
    
    % 获取所有场景目录
    all_dirs = dir(data_path);
    all_dirs = all_dirs([all_dirs.isdir]);
    all_dirs = all_dirs(~ismember({all_dirs.name}, {'.', '..'}));
    
    if scene_idx > length(all_dirs)
        error('Scene index %d exceeds available scenes (%d)', scene_idx, length(all_dirs));
    end
    
    % 选择场景
    scene_name = all_dirs(scene_idx).name;
    scene_path = fullfile(data_path, scene_name, scene_name);
    
    fprintf('Loading scene: %s\n', scene_name);
    
    % 读取场景数据
    filelist = dir(fullfile(scene_path, '*.png'));
    if isempty(filelist)
        error('No PNG files found in %s', scene_path);
    end
    
    % 初始化数据
    img_clean = zeros(512, 512, 31);
    
    % 读取31个光谱通道
    for j = 1:31
        if j <= length(filelist)
            img_path = fullfile(scene_path, filelist(j).name);
            im = imread(img_path);
            img_clean(:,:,j) = im2double(im(:,:,1));
        end
    end
    
    % 加载模糊核
    kernel_file = fullfile(kernel_path, sprintf('kernel_%d.mat', kernel_idx));
    if exist(kernel_file, 'file')
        kernel_data = load(kernel_file);
        kernel = kernel_data.kernel;
    else
        % 如果核文件不存在，生成一个高斯核
        fprintf('Warning: Kernel file not found, using Gaussian kernel\n');
        kernel = fspecial('gaussian', 5, 1);
    end
    
    % 应用模糊和噪声
    noise_level = (kernel_idx == 3) * 0.03 + (kernel_idx ~= 3) * 0.01;
    
    img_blurred = zeros(size(img_clean));
    for i = 1:31
        img_blurred(:,:,i) = imfilter(img_clean(:,:,i), kernel, 'circular', 'conv');
    end
    img_blurred = img_blurred + noise_level * randn(size(img_blurred));
    
    % 确保值在[0,1]范围内
    img_blurred = max(0, min(1, img_blurred));
    
    fprintf('Data loaded: size = [%d, %d, %d], noise level = %.3f\n', ...
        size(img_clean), noise_level);
end