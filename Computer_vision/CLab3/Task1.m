% CLAB3 Task1
img_left = imread('CLab3_Materiala/Left.jpg');
img_right = imread('CLab3_Materiala/Right.jpg');

% imshow(I);
subplot(1,2,1)
imshow(img_left)
subplot(1,2,2)
imshow(img_right)
display('click mouse for 6 features...')
uv = ginput(12);    % Graphical user interface to get 12 points,
% 6 on left and 6 on right
display(uv);
%% 
uBase = uv(1:6, 1);
vBase = uv(1:6, 2);

u2Trans = uv(7:12, 1);
v2Trans = uv(7:12, 2);

H = homography(u2Trans, v2Trans, uBase, vBase)
im_left_wrap = ImageWarping(img_left, H)
imshow(im_left_wrap)


%% 
function im_warped = ImageWarping(im, H)
	[u_x, u_y] = GetPointsToTransform(size(im,2), size(im,1));
	[v_x, v_y] = TransformPointsUsingHomography(inv(H), u_x, u_y);
	im_warped = BuildWarpedImage(double(im), v_x, v_y);
	im_warped = uint8(im_warped);
end

function [u_x, u_y] = GetPointsToTransform(width, height)
	[u_x, u_y] = meshgrid(1:width, 1:height);
end

function [v_x, v_y] = TransformPointsUsingHomography(H, u_x, u_y)
	v_x = H(1,1)*u_x + H(1,2)*u_y + H(1,3);
	v_y = H(2,1)*u_x + H(2,2)*u_y + H(2,3);
	v_z = H(3,1)*u_x + H(3,2)*u_y + H(3,3);

	v_x = v_x./v_z;
	v_y = v_y./v_z;
end

function im_warped = BuildWarpedImage(im, v_x, v_y)
	h = size(v_x, 1); w = size(v_x,2);
	im_warped(:,:,1) = reshape(interp2(im(:,:,1), v_x(:), v_y(:)), [h, w]);
	im_warped(:,:,2) = reshape(interp2(im(:,:,2), v_x(:), v_y(:)), [h, w]);
	im_warped(:,:,3) = reshape(interp2(im(:,:,3), v_x(:), v_y(:)), [h, w]);
end



% Attempt to normalise each set of points so that the origin 
% is at centroid and mean distance from origin is sqrt(2).
% [x1, T1] = normalise2dpts(x1);
% [x2, T2] = normalise2dpts(x2);
% Note that it may have not been possible to normalise
% the points if one was at infinity so the following does not
% assume that scale parameter w = 1.
%% TASK 1: 
function H = homography(u2Trans, v2Trans, uBase, vBase)
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = H p' , i.e.,:  
% (uBase, vBase, 1)'=H*(u2Trans , v2Trans, 1)' 
% 
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
unit=ones(6,1);
% image coordinates in the Base image planar
p1=[uBase vBase unit]';
% image coordinates in the Trans image planar
p2=[u2Trans v2Trans unit]';

[p1_norm, p1_trans]= normalise2dpts(p1);
[p2_norm, p2_trans]= normalise2dpts(p2);


% to compute the matrix H, we should construct matrix A
A=[];
tempA=[];
% for every point, we have only two linearly independent equations
% I combine them together and get a 12x9 matrix A
for i=1:6
    tempA=[zeros(1,3) (-1)*p2_norm(:,i)' p1_norm(2,i)*p2_norm(:,i)';
        p2_norm(:,i)' zeros(1,3) -p1_norm(1,i)*p2_norm(:,i)'];
    A=[A ;tempA];
end

% conduct SVD on matrix A
[U,S,V]=svd(A);
% the right-most-column of V is the h with least squares
h=V(:,end);
H=reshape(h,[3, 3])';

% denormalise H, according to the transformation matrixs
H = inv(p1_trans)*H*(p2_trans);
end

% NORMALISE2DPTS - normalises 2D homogeneous points
%
% Function translates and normalises a set of 2D homogeneous points 
% so that their centroid is at the origin and their mean distance from 
% the origin is sqrt(2).  This process typically improves the
% conditioning of any equations used to solve homographies, fundamental
% matrices etc.
%
% Usage:   [newpts, T] = normalise2dpts(pts)
%
% Argument:
%   pts -  3xN array of 2D homogeneous coordinates
%
% Returns:
%   newpts -  3xN array of transformed 2D homogeneous coordinates.  The
%             scaling parameter is normalised to 1 unless the point is at
%             infinity. 
%   T      -  The 3x3 transformation matrix, newpts = T*pts
%           
% If there are some points at infinity the normalisation transform
% is calculated using just the finite points.  Being a scaling and
% translating transform this will not affect the points at infinity.

function [newpts, T] = normalise2dpts(pts)

    if size(pts,1) ~= 3
        error('pts must be 3xN');
    end
    
    % Find the indices of the points that are not at infinity
    finiteind = find(abs(pts(3,:)) > eps);
    
    if length(finiteind) ~= size(pts,2)
        warning('Some points are at infinity');
    end
    
    % For the finite points ensure homogeneous coords have scale of 1
    pts(1,finiteind) = pts(1,finiteind)./pts(3,finiteind);
    pts(2,finiteind) = pts(2,finiteind)./pts(3,finiteind);
    pts(3,finiteind) = 1;
    
    c = mean(pts(1:2,finiteind)')';            % Centroid of finite points
    newp(1,finiteind) = pts(1,finiteind)-c(1); % Shift origin to centroid.
    newp(2,finiteind) = pts(2,finiteind)-c(2);
    
    dist = sqrt(newp(1,finiteind).^2 + newp(2,finiteind).^2);
    meandist = mean(dist(:));  % Ensure dist is a column vector for Octave 3.0.1
    
    scale = sqrt(2)/meandist;
    
    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];
    
    newpts = T*pts;
end

function C = calibrate(im, XYZ, uv)
%% TASK 2: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   K = calibrate(image, XYZ, uv)
%
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.

end




    
function [R,Q] = rq(A)
%% RQ factorisation

[q,r]=qr(A');       % Matlab has QR decomposition but not RQ decomposition  
                    % with Q: orthonormal and R: upper triangle. Apply QR
                    % for the A-transpose, then A = (qr)^T = r^T*q^T = RQ
R=r';
Q=q';
end
