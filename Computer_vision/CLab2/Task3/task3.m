%%
folder = 'CLab2/Task3/Yale-FaceA/my_front/';
folder_save = 'CLab2/Task3/Yale-FaceA/my_front_crop/';
imagefiles = dir(strcat(folder,'*.jpg'));  
nfiles = length(imagefiles);    % Number of files found
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(strcat(folder,currentfilename));
   rect = [500,731,695,926];
   currentimage = imgcrop(currentimage, rect);
   imgwrite(strcat(folder_save,i));
end
%% load train files
folder = 'CLab2/Task3/Yale-FaceA/trainingset/';
imagefiles = dir(strcat(folder,'*.png'));  
nfiles = length(imagefiles);    % Number of files found
imgary_train = [];
for i=1:nfiles
   currentfilename = imagefiles(i).name;
   currentimage = imread(strcat(folder,currentfilename));
   images_train{i} = currentimage;   % the cell of high dim img array
   [imgsize_r, imgsize_c] = size(currentimage);
   temp = reshape(currentimage',imgsize_r*imgsize_c,1); % Reshaping 2D images into 1D image vectors
                                % here img' is used because reshape(A,M,N) function reads the matrix A columnwise
                                % where as an image matrix is constructed with first N pixels as first row,next N in second row so on
   imgary_train = [imgary_train temp];      % X,the image matrix with columnsgetting added for each image
%    imgary = cat(2, imgary, images{ii});
end
% imwrite(imgary, 'test_imgary.png');
% apply PCA
% imgary = im2double(imgary);
% coeff = pca(imgary);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   calculate m, A and eigenfaces
%
%          m           -    (MxN)x1  Mean of the training images
%          A           -    (MxN)xP  Matrix of image vectors after each vector getting subtracted from the mean vector m
%     eigenfaces       -    (MxN)xP' P' Eigenvectors of Covariance matrix (C) of training database X
%                                    where P' is the number of eigenvalues of C that best represent the feature set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mean image vector
mean_img_vct = mean(imgary_train,2); 

%% 
tmimg=uint8(mean_img_vct);   %converts to unsigned 8-bit integer. Values range from 0 to 255
img=reshape(tmimg,imgsize_c,imgsize_r);    %takes the N1*N2x1 vector and creates a N2xN1 matrix
img=img';       %creates a N1xN2 matrix by transposing the image.
figure(3);
imshow(img);
%%
% Computing the average face image m = (1/P)*sum(Xj's)    (j = 1 : P)
imgcount = size(imgary_train,2);
% A matrix, after subtraction of all image vectors from the mean image vector
DevMtx = [];
for i=1 : imgcount
    temp = double(imgary_train(:,i)) - mean_img_vct;
    DevMtx = [DevMtx temp];
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATION OF EIGENFACES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  For a MxN matrix, the maximum number of non-zero eigenvalues that its covariance matrix can have
%%%  is min[M-1,N-1]. 
%%%  The number of dimensions (pixels) of each image vector is very high compared to number of
%%%  test images here, so number of non-zero eigenvalues of C will be maximum P-1 (P being the number of test images)
%%%  if we calculate eigenvalues & eigenvectors of C = A*A' , then it will be very time consuming as well as memory.
%%%  so we calculate eigenvalues & eigenvectors of L = A'*A , whose eigenvectors will be linearly related to eigenvectors of C.
%%%  these eigenvectors being calculated from non-zero eigenvalues of C, will represent the best feature sets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L= DevMtx' * DevMtx;
[V,D]=eig(L);  %% V : eigenvector matrix  D : eigenvalue matrix
%%%% again we use Kaiser's rule here to find how many Principal Components (eigenvectors) to be taken
%%%% if corresponding eigenvalue is greater than 1, then the eigenvector will be chosen for creating eigenface
L_eig_vec = [];
for i = 1 : size(V,2) 
    if( D(i,i) > 1 )
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end
%%% finally the eigenfaces %%%
eigenfaces = DevMtx * L_eig_vec;

%% display top eigenfaces
k_pc = 10;
ef = [];
for i = 1:k_pc
  subplot(2,ceil(k_pc/2),i);
  temp = reshape(eigenfaces(:,i),imgsize_r,imgsize_c);
  temp = temp';
%   temp = histeq(temp,255);
  ef = [ef temp];
  imshow(temp, [])
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recognition, compare three faces by projecting the images into facespace and 
% measuring the Euclidean distance between them using nearest neighbour
%
%            recogimg           -   the recognized image name
%             testimg           -   the path of test image
%                m              -   mean image vector
%                A              -   mean subtracted image vector matrix
%           eigenfaces          -   eigenfaces that are calculated from eigenface function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% finding the projection of each image vector on the facespace 
% (where the eigenfaces are the co-ordinates or dimensions)
projectimg = [ ];  % projected image vector matrix
for i = 1 : size(eigenfaces,2)
    temp = eigenfaces' * DevMtx(:,i);
    projectimg = [projectimg temp];
end

%% load test files
folder = 'CLab2/Task3/Yale-FaceA/testset/';
% folder = 'CLab2/Task3/Yale-FaceA/my_front_cut/';
imagefiles = dir(strcat(folder,'*.png'));  
nfiles = length(imagefiles);    % Number of files found
imgary_test = [];
for i=1:nfiles
    currentfilename = imagefiles(i).name;
    currentimage = imread(strcat(folder,currentfilename));
    images_test{i} = currentimage;   % the cell of high dim img array
    [imgsize_r, imgsize_c] = size(currentimage);
    temp = reshape(currentimage',imgsize_r*imgsize_c,1); 
    % Reshaping 2D images into 1D image vectors
    % here img' is used because reshape(A,M,N) function reads the matrix A columnwise
    % where as an image matrix is constructed with first N pixels as first row,next N in second row so on
    imgary_test = [imgary_test temp];      % X,the image matrix with columnsgetting added for each image
%    imgary = cat(2, imgary, images{ii});
end
%% extractiing PCA features of the test image %%%%%
test_image = imgary_test(1);
test_image = test_image(:,:,1);
[imgsize_r imgsize_c] = size(test_image);
temp = reshape(test_image',imgsize_r*imgsize_c,1); 
% creating (MxN)x1 image vector from the 2D image
temp = double(temp)-mean_img_vct; % mean subtracted vector
projtestimg = eigenfaces'*temp; % projection of test image onto the facespace
%%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
euclide_dist = [ ];
for i=1 : size(eigenfaces,2)
    temp = (norm(projtestimg-projectimg(:,i)))^2;
    euclide_dist = [euclide_dist temp];
end
[euclide_dist_min recognized_index] = min(euclide_dist);
recognized_img = strcat(int2str(recognized_index),'.jpg');
%% plot
for i = 1:3
subplot(1,3,i);
imshow(images_train{recognized_index});
end