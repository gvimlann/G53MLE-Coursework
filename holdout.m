function [htrain,htest] = holdout(data,test_ratio)
    if ~isscalar(data)
        data = length(data);
    end
    if ~isa(test_ratio, 'numeric')
        disp('Test data ratio must be between 0 and 1');
        return;
    end
    
    rd = round(data * test_ratio);
    htrain = zeros([data 1]);
    htest = zeros([data 1]);
    test_perm_idx = randperm(data, rd)';
    
    for i = 1:rd
        htest(test_perm_idx(i)) = true;
    end
    htest = logical(htest);
    htrain = ~htest;
        
end

