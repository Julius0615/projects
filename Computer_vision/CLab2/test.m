%%
% CLAB2 Task-1: Harris Corner Detector
% You Li u6430173
%

task_image =  imread('CLab2/Task1/Harris_1.jpg');

% frame = rgb2gray(frame)
I = double(task_image);
% points = detectHarrisFeatures(I)
% imshow(I); hold on;
% plot(corners.selectStrongest(50));
%****************************
% imshow(task_image);
min_N=12;max_N=100;
%%%%%%%%%%%%%%Intrest Points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigma=2; Thrshold=0.01; r=11; disp=0;
dx = [-1 0 1; -1 0 1; -1 0 1]; % The Mask
dy = dx';
% compute x and y derivatives of image

%%%%%%
testArr = rgb2gray(task_image)
Ix = conv2(testArr, dx, 'same');
Iy = conv2(testArr, dy, 'same');

g = fspecial('gaussian',max(1,fix(6*sigma)), sigma); %%%%%% Gaussien Filter


%%%%%
Ix2 = conv2(Ix.^2, g, 'same');
Iy2 = conv2(Iy.^2, g, 'same');
Ixy = conv2(Ix.*Iy, g,'same');
%%%%%%%%%%%%%%
k = 0.04;
R11 = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;
R11=(1000/max(max(R11)))*R11;
R=R11;
ma=max(max(R));
sze = 2*r+1;
MX = ordfilt2(R,sze^2,ones(sze));
R11 = (R==MX)&(R>Thrshold);
count=sum(sum(R11(5:size(R11,1)-5,5:size(R11,2)-5)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Task: Perform non-maximum suppression and             %
%       thresholding, return the N corner points        %
%       as an Nx2 matrix of x and y coordinates         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loop=0;
while (((count<min_N)|(count>max_N))&(loop<30))
    if count>max_N
        Thrshold=Thrshold*1.5;
    elseif count < min_N
        Thrshold=Thrshold*0.5;
    end
    
    R11 = (R==MX)&(R>Thrshold);
    count=sum(sum(R11(5:size(R11,1)-5,5:size(R11,2)-5)));
    loop=loop+1;
end


R=R*0;
R(5:size(R11,1)-5,5:size(R11,2)-5)=R11(5:size(R11,1)-5,5:size(R11,2)-5);
[r1,c1] = find(R);
H_corners=[r1+cmin,c1+rmin]%% IP

%%%%%%%%%%%%%%%%%%%% Display

Size_PI=size(H_corners,1);
for r=1: Size_PI
    I(H_corners(r,1)-2:H_corners(r,1)+2,H_corners(r,2)-2)=255;
    I(H_corners(r,1)-2:H_corners(r,1)+2,H_corners(r,2)+2)=255;
    I(H_corners(r,1)-2,H_corners(r,2)-2:H_corners(r,2)+2)=255;
    I(H_corners(r,1)+2,H_corners(r,2)-2:H_corners(r,2)+2)=255;
    
end

imshow(uint8(I))


