function model = SVM(classification_type,X,Y)

    if strcmpi(classification_type, 'linear_classification')
        model = fitcsvm(X,Y, 'KernelFunction','linear', 'BoxConstraint',1);
    
    elseif strcmpi(classification_type, 'linear_regression')
        model = fitrsvm(X,Y, 'KernelFunction','linear', 'BoxConstraint',1, 'Epsilon', 1);
        
    elseif strcmpi(classification_type, 'polynomial_classification')
        model = fitcsvm(X,Y, 'KernelFunction','polynomial', 'PolynomialOrder', q, 'BoxConstraint',1);
        
    elseif strcmpi(classification_type, 'polynomial_regression')
        model = fitrsvm(X,Y, 'KernelFunction','polynomial', 'PolynomialOrder', q, 'BoxConstraint',1, 'Epsilon', 1);
    
    elseif strcmpi(classification_type, 'rbf_classification')
        model = fitcsvm(X,Y, 'KernelFunction', 'rbf', 'KernelScale', sigma, 'BoxConstraint',1);
        
    elseif strcmpi(classification_type, 'rbf_regression')
        model = fitrsvm(X,Y, 'KernelFunction','rbf', 'KernelScale', sigma, 'BoxConstraint',1, 'Epsilon', 1);
        
    end
end