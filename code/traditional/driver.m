beginIndex = 184;
endIndex = beginIndex;

write = 1;
display = 1;

thresholding = 1;
k_means = 1;

root = fullfile('/home/albert/Documents/Classes/CS670/CourseProject/melanoma_classification');
dataDir = fullfile(root,'data','test');
testDir = fullfile(root,'data','ground_truth');
outDir = fullfile(root,'data','out');

imagesPattern = fullfile(dataDir, '*.jpg');
images = dir(imagesPattern);
numImages = length(images);

scores = zeros(numImages,6*thresholding+k_means);

for i = beginIndex:endIndex
    imageName = images(i).name(1:end-4);
    thisImage = fullfile(dataDir, strcat(imageName, '.jpg'));
    im = imread(thisImage);
    gt = imread(fullfile(testDir, strcat(imageName, '_segmentation.png')));
    if thresholding
        outs = illuminati(im);
        [h,w,n] = size(outs);
        for out = 1:n
            if write
                writeOut = outs(:,:,out);
                suffix = sprintf('_thresh%d', out);
                imwrite(writeOut, fullfile(outDir, strcat(imageName, suffix, '.png')));
            end

            if display
                subplot(n+k_means,2,out*2-1), imshow(gt);
                subplot(n+k_means,2,out*2), imshow(outs(:,:,out));
            end

            scores(i+1-beginIndex,out) = scores(i+1-beginIndex,out) + testrep(outs(:,:,out), gt);
        end
    end
    if k_means
        outs = kaymeans(im);
        if ~thresholding
            [h, w, ~] = size(outs);
            n = 0;
        end
        if write
            writeOut = outs(:,:,1);
            imwrite(writeOut, fullfile(outDir, strcat(imageName, '_kmeans.png')));
        end
        if display
            subplot(n*thresholding+1,2,n*2+1), imshow(gt);
            subplot(n*thresholding+1,2,n*2+2), imshow(outs(:,:,1));
        end
        
        scores(i+1-beginIndex,n+1) = scores(i+1-beginIndex,n+1) + testrep(outs(:,:,1), gt);
    end        
end
scores = sum(scores, 1);
disp(scores);