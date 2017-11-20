function [xtrain,ytrain,xtest,ytest,trainInd,testInd] = holdout(x, y, test_ratio)
    if ~isscalar(x)
        szx = size(x, 2);
    end
    if ~isa(test_ratio, 'numeric')
        disp('Test data ratio must be between 0 and 1');
        return;
    end
    
    rd = round(szx * test_ratio);
%     trainInd = zeros([x 1]);
    testInd = zeros([szx 1]);
    test_perm_idx = randperm(szx, rd)';
    
    for i = 1:rd
        testInd(test_perm_idx(i)) = true;
    end
    testInd = logical(testInd);
    trainInd = ~testInd;
    
    xtrain = x(:, trainInd);
    ytrain = y(:, trainInd);
    xtest = x(:, testInd);
    ytest = y(:, testInd);
end

