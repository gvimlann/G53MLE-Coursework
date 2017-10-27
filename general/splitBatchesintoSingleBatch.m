% Author: VIMLAN G
% parameters(index, training_batches, validation_batches) - returns
% training batch and validation batch based on index
function [points_batch,points_label] = splitBatchesintoSingleBatch(points_batches,points_labels,index)
    points_batch = points_batches(index,:,:);
    points_batch = reshape(points_batch,[size(points_batch,2),size(points_batch,3)]);
    points_label = points_labels(index,:);
    points_label = reshape(points_label,[1,size(points_label,2)]);
end