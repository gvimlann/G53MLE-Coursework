function gainOut = gain(labels,remainder)
    gainOut = calculate_entropy(labels) - remainder;
end