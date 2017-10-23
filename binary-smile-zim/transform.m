function data = transform(points, type)
    data = zeros(size(points, 1)*size(points, 2), size(points, 3));
    
    %Reposition the data    
    if (type == 1)
        % Vimlan's way
        data(1:66, :) = points(:, 1, :);
        data(67:end, :) = points(:, 2, :);
    elseif (type == 2)
        % Columns = Samples
        % Row = Coordinates (X followed by Y)
        % |x1|  |
        % |y1|  |
        % |x2|  |
        % |y2|  |
        data(1:2:end, :) = points(:, 1, :);
        data(2:2:end, :) = points(:, 2, :);
    end    
end
