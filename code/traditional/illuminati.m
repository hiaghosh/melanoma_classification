function segmented = illuminati(im)

[h, w, n] = size(im);

representations = {@(x)x, @rgb2hsv};

[~, reps] = size(representations);

segmented = zeros(h, w , n*reps);

for rep = 1:reps
    image = representations{rep}(im);
    for channel = 1:n
        segmented(:,:,(rep-1)*n+channel) = bwareaopen(imcomplement(im2bw(imadjust(image(:,:,channel)))), 1000);
    end
end
        
    
        
 
