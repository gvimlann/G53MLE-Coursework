% @author Boey
% Segments data into k folds by returning indices carrying value for each
% fold
% Vector segmentation is done using random permutations to ensure
% randomness of indices
function kindices = kfoldcross(data, k, use_randperm)
    if ~isscalar(data)
        data = size(data, 2);
    end
    if ~isscalar(k)
        disp('K-value must be scalar');
        return;
    end
    
    kindices = zeros([data 1]);
    perm_idx = randperm(data)';
    idx = repmat((1:1:k)', round(data / k), 1);
    
    if length(idx) < data
        for i = length(idx) + 1:1:data
            idx(i) = randi(k);
        end
    elseif length(idx) > data
        idx = idx(1:data);
    end
    
    if use_randperm
        for i = 1:length(kindices)
            kindices(perm_idx(i)) = idx(i);
        end
    else
        kindices = idx;
    end
    
end

