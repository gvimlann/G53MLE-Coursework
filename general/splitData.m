% Author: VIMLAN G
% data - training data, percentageSplit - percentage to split into train
% and remaining into test
function [train,test] = splitData(data,percentageSplit)
    percentageSplit = percentageSplit/100;
    train = data(1:floor(percentageSplit*data),:);
    test = data(floor(percentageSplit*data)+1:size(data,1),:);
end
