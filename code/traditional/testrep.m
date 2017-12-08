function score = testrep( segmented, im )

[h,w] = size(im);

score = pdist2(im, segmented, 'hamming');

score = sum(sum(score))/(h*w);

end

