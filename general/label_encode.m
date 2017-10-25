function vec_encode = label_encode(label)
%VECTORISE_ENCODE Encode value represented labels into vector of 0s with
%flag 1 where the index of the flag is the value of the label

    if length(size(label, 2)) > 2 || size(label, 2) > 1
        disp('No support for multi column labels and 2-dimentional datasets');
        return;
    end

    vec_encode = zeros([length(label) max(label)]);

    for i = 1:length(label)
        vec_encode(i, label(i, 1)) = 1;
    end

end

