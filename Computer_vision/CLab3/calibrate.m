function [mse,C] = calibrate(im, XYZ, uv)
%GET_CAMERA_MATRIX Summary of this function goes here
%   Detailed explanation goes here
homo_X = inhomogeneous2homogeneous(XYZ)
homo_x = inhomogeneous2homogeneous(uv)
A = get_A(homo_x,homo_X)
[U,D,V] = svd(A,0); 
C = reshape(V(:,12),4,3)';
% C(4,3)=1;
C
project_x = zeros(12, 2, 'double');
for i=1: 12
    % Project to image coordinates
    result = C * homo_X(:,i);
    result = result / result(3);
    project_x(i, :) = result(1:2, :);
end

% Decompose the C matrix into K, R, t
[K, R, t] = vgg_KR_from_P(C);
K
R
t


homo_coordinates_system = inhomogeneous2homogeneous([
    0 0 0;  % origin
    3 0 0;  % X
    0 3 0;  % Y
    0 0 3;  % Z
]*7);
direction = ['0', 'X', 'Y', 'Z'];
% Pre-allocate a matrix for the points
homo_system = zeros(4, 2, 'double');
for i=1: 4
    % Project to image coordinates
    result = C * homo_coordinates_system(:,i);
    result = result / result(3);
    homo_system(i, :) = result(1:2, :);
end


% Compute the pitch angle of the camera with respect to the X-Z plane 
% in the world coordinate system
theta_y = atan2(-R(3, 1), sqrt(R(3, 2)^2 + R(3, 3)^2));
mse = sum((project_x - uv).^2, 'all') / 12;
imshow(im),colormap(gray), hold on
plot(uv(:,1),uv(:,2),'rx'),plot(project_x(:, 1), project_x(:, 2), 'ob', 'MarkerSize', 8),
fprintf('MSE %.2f \n',mse);
fprintf('The pitch angle is: %.2f degrees\n', theta_y * (180 / pi));


for i =1:4
    % Draw the line, arrow and text for x y z 
    quiver(homo_system(1, 1), homo_system(1, 2), homo_system(i, 1) - homo_system(1, 1), homo_system(i, 2) - homo_system(1, 2), 1, 'LineWidth', 2);
    text(homo_system(i, 1), homo_system(i, 2), direction(i), 'Color', 'b', 'FontSize', 14);
end
hold off;


end

