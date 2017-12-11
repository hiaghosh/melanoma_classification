root = fullfile('/home/albert/Documents/Classes/CS670/CourseProject/melanoma_classification');
dataDir = fullfile(root,'data','test');
testDir = fullfile(root,'data','out');
outDir = fullfile(root,'data');

filename = 'ISIC_0000483';

im1 = imread(fullfile(dataDir, strcat(filename, '.jpg')));

im2 = imread(fullfile(testDir, strcat(filename, '_segmented.png')));

imshow(im1);

imwrite(bsxfun(@times, im1, cast(repmat(im2bw(im2),1,1,3), 'like', im1)),fullfile(outDir, strcat(filename, '.jpg')));