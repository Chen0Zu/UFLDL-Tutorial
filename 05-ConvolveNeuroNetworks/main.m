clc;clear;
addpath(genpath('../minFunc_2012'));
addpath('../common');
rng(1);


%%
function f = sigmoid(x)
f = 1./(1+exp(-x));
end

function convFeature = cnnConvolve(filterDim, nFilters,images, W, b)
nImages = size(images,3);
imageDim = size(images,1);
convDim = imageDim - filterDim + 1;

convFeature = zeros(convDim, convDim, nFilters, nImages);

for iImage = 1:nImages
    for iFilter = 1:nFilters
        image = images(:,:,iImage);
        filter = W(:,:,iFilter);
        % Flip W for use in conv2
        filter = rot90(filter,2);
        convImage = conv2(image,filter,'valid');
        convFeature(:,:,iFilter,iImage) = ...
            sigmoid(convImage + b(iFilter));
    end
end

end