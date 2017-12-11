function segmented = kaymeans(im)

%convert image to L*a*b color space
cform = makecform('srgb2lab');
lab_he = applycform(im,cform);

%classify the colors in 'a*b*' Space Using K-Means Clustering
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);

%Label Every Pixel in the Image Using the Results from KMEANS
pixel_labels = reshape(cluster_idx,nrows,ncols);
%imshow(pixel_labels,[]), title('image labeled by cluster index');

segmented(:,:,1) = im2bw(imadjust(pixel_labels-1));
