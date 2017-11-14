function remainderOut = remainder(leftLabels,rightLabels)
    totalLabels = size(leftLabels,1) + size(rightLabels,1);
    leftLabelSize = size(leftLabels,1);
    rightLabelSize = size(rightLabels,1);
    
    entropyLeft = calculate_entropy(leftLabels);
    entropyRight = calculate_entropy(rightLabels);
    remainderOut = leftLabelSize/totalLabels * entropyLeft + rightLabelSize/totalLabels * entropyRight;
end