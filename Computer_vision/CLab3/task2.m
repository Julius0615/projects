% CLAB3 Task 2
img_a = imread('CLab3_Materiala/stereo2012a.jpg');
img_b = imread('CLab3_Materiala/stereo2012b.jpg');
img_c = imread('CLab3_Materiala/stereo2012c.jpg');
img_d = imread('CLab3_Materiala/stereo2012d.jpg');

%store cooridates.
XYZ = [7,7,0;
        7,0,7;
        0,7,7;
        21,14,0;
        0,21,14;
        14,0,21;
        14,7,0;
        7,0,14;
        0,14,7;
        21,21,0;
        21,0,21;
        0,21,21];
imshow(img_a);
hold on;
[u,v] = ginput(12);
plot(u,v,'ro');

save data.mat XYZ u v; 

display('click mouse for 6 features...')
% uv = ginput(6);    % Graphical user interface to get 12 points
display(uv);
plot([origin_2D(1,1),X_2D(1,1)],[origin_2D(2,1),X_2D(2,1)],'Color','r','LineWidth',1);
plot([origin_2D(1,1),Y_2D(1,1)],[origin_2D(2,1),Y_2D(2,1)],'Color','r','LineWidth',1);
plot([origin_2D(1,1),Z_2D(1,1)],[origin_2D(2,1),Z_2D(2,1)],'Color','r','LineWidth',1);

%% 
% C = calibrate_1(img_a,XYZ,[u,v]);
C = calibrate(img_a,XYZ,[u,v]);
disp(C);

%% 
[K, R, t] = vgg_KR_from_P(C);
disp('K = ');
disp(K);
disp('R = ');
disp(R);
disp('t = ');
disp(t);
%calculate focal
alpha = K(1,1);
gama = K(1,2);
thelta = acot(gama/-alpha);
fx = abs(alpha);
fy = abs(K(2,2)*sin(thelta));
f = sqrt(fx^2+fy^2);
disp(['fx: ',num2str(fx)]);
disp(['fy: ',num2str(fy)]);
disp(['focal: ',num2str(f)]);

%% 
function C = calibrate(im, XYZ, uv)
[len,~] = size(uv);

A = zeros(len*2,12);

for tmp = 1:len
   %get the 3D vertices and their 2D corresponding points
   X = XYZ(tmp,1);
   Y = XYZ(tmp,2);
   Z = XYZ(tmp,3);
   
   u = uv(tmp,1);
   v = uv(tmp,2);
   
   A(tmp*2-1,:) = [X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z,-u];
   A(tmp*2,:) = [0,0,0,0,X,Y,Z,1,-v*X,-v*Y,-v*Z,-v];
end
[~,~,V] = svd(A);

C = V(:,end);
C = C/C(end);
C = reshape(C,[4 3]);
C = C';

imshow(im);
hold on;
plot(uv(:,1),uv(:,2),'ro');
hold on;
xyz1 = [XYZ,ones(tmp,1)]';
xy = C*xyz1;

xy(1,:) = xy(1,:)./xy(3,:);
xy(2,:) = xy(2,:)./xy(3,:);

distance = (xy(1:2,:) - uv').^2;
distance = mean(sum(distance));
disp(['Mean Square Error: ',num2str(distance)]);

plot(xy(1,:),xy(2,:),'g+');

end


%% TASK 2: CALIBRATE
function C = calibrate_1(im, XYZ, uv)
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
[len,~] = size(uv);

A = zeros(len*2,12);

for tmp = 1:len
   %get the 3D vertices and their 2D corresponding points
   X = XYZ(tmp,1);
   Y = XYZ(tmp,2);
   Z = XYZ(tmp,3);
   
   u = uv(tmp,1);
   v = uv(tmp,2);
   
   A(tmp*2-1,:) = [X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z,-u];
   A(tmp*2,:) = [0,0,0,0,X,Y,Z,1,-v*X,-v*Y,-v*Z,-v];
end
[~,~,V] = svd(A);

C = V(:,end);
C = C/C(end);
C = reshape(C,[4 3]);
C = C';

imshow(im);
hold on;
plot(uv(:,1),uv(:,2),'ro');
hold on;
xyz1 = [XYZ,ones(tmp,1)]';
xy = C*xyz1;

xy(1,:) = xy(1,:)./xy(3,:);
xy(2,:) = xy(2,:)./xy(3,:);

distance = (xy(1:2,:) - uv').^2;
distance = mean(sum(distance));
disp(['Mean Square Error: ',num2str(distance)]);

plot(xy(1,:),xy(2,:),'g+');


end

function [R,Q] = rq(A)
%%RQ factorisation

[q,r]=qr(A');       % Matlab has QR decomposition but not RQ decomposition  
                    % with Q: orthonormal and R: upper triangle. Apply QR
                    % for the A-transpose, then A = (qr)^T = r^T*q^T = RQ
R=r';
Q=q';
end

%VGG_KR_FROM_P Extract K, R from camera matrix.
%
%    [K,R,t] = VGG_KR_FROM_P(P [,noscale]) finds K, R, t such that P = K*R*[eye(3) -t].
%    It is det(R)==1.
%    K is scaled so that K(3,3)==1 and K(1,1)>0. Optional parameter noscale prevents this.
%
%    Works also generally for any P of size N-by-(N+1).
%    Works also for P of size N-by-N, then t is not computed.


% Author: Andrew Fitzgibbon <awf@robots.ox.ac.uk>
% Modified by werner.
% Date: 15 May 98


function [K, R, t] = vgg_KR_from_P(P)

N = size(P,1);
H = P(:,1:N);

[K,R] = vgg_rq(H);
  
if nargin < 2
  K = K / K(N,N);
  if K(1,1) < 0
    D = diag([-1 -1 ones(1,N-2)]);
    K = K * D;
    R = D * R;
    
  %  test = K*R; 
  %  vgg_assert0(test/test(1,1) - H/H(1,1), 1e-07)
  end
end

if nargout > 2
  t = -P(:,1:N)\P(:,end);
end


end

%% 
% [R,Q] = vgg_rq(S)  Just like qr but the other way around.
%
% If [R,Q] = vgg_rq(X), then R is upper-triangular, Q is orthogonal, and X==R*Q.
% Moreover, if S is a real matrix, then det(Q)>0.

% By awf

function [U,Q] = vgg_rq(S)

S = S';
[Q,U] = qr(S(end:-1:1,end:-1:1));
Q = Q';
Q = Q(end:-1:1,end:-1:1);
U = U';
U = U(end:-1:1,end:-1:1);

if det(Q)<0
  U(:,1) = -U(:,1);
  Q(1,:) = -Q(1,:);
end

end

function A = get_A(x,X)
%GET_A Summary of this function goes here
%   Detailed explanation goes here
n = size(x, 2);
h = 1;
for k =1:n
    A(h, :) = [X(1, k) X(2, k) X(3, k) 1 ...
        0 0 0 0 ...
        -x(1, k)*X(1, k) -x(1, k)*X(2, k) ...
        -x(1, k)*X(3, k) -x(1, k)];
    A(h + 1, :) = [0 0 0 0 ...
         X(1, k) X(2, k) X(3, k) 1 ...
          -x(2, k)*X(1, k) -x(2, k)*X(2, k) ...
          -x(2, k)*X(3, k) -x(2, k)];
    h = h +2;
end

end


