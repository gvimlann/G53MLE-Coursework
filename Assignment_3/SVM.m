function model = SVM(X, Y, classification_type, varargin)
    p = inputParser;
    
    defaultKernalScale = 1;
    defaultPolynomialOrder = 3;
    defaultEpsilon = iqr(Y) / 13.49;
    
    addParameter(p, 'KernelScale', defaultKernalScale, @isnumeric);
    addParameter(p, 'PolynomialOrder', defaultPolynomialOrder, @isnumeric);
    addParameter(p, 'Epsilon', defaultEpsilon, @isnumeric);
    parse(p, varargin{:});
    
    if strcmpi(classification_type, 'linear_classification')
        model = fitcsvm(X, Y, 'KernelFunction', 'linear', 'BoxConstraint', 1);
    
    elseif strcmpi(classification_type, 'linear_regression')
        model = fitrsvm(X, Y, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Epsilon', p.Results.Epsilon);
        
    elseif strcmpi(classification_type, 'polynomial_classification')
        model = fitcsvm(X, Y, 'KernelFunction', 'polynomial', 'PolynomialOrder', p.Results.PolynomialOrder, 'BoxConstraint', 1);
        
    elseif strcmpi(classification_type, 'polynomial_regression')
        model = fitrsvm(X, Y, 'KernelFunction', 'polynomial', 'PolynomialOrder', p.Results.PolynomialOrder, 'BoxConstraint', 1, 'Epsilon', p.Results.Epsilon);   
    
    elseif strcmpi(classification_type, 'rbf_classification')
        model = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'KernelScale', p.Results.KernelScale, 'BoxConstraint', 1);
        
    elseif strcmpi(classification_type, 'rbf_regression')
        model = fitrsvm(X, Y, 'KernelFunction', 'rbf', 'KernelScale', p.Results.KernelScale, 'BoxConstraint', 1, 'Epsilon', p.Results.Epsilon);
        
    end
end