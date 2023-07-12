using Images, FileIO, LinearAlgebra, Colors

function get_row_column(img_path)
    img = load(img_path)
    row,column = size(img)
    return (row,column)
end

function image_to_vector(img_path)
    img = load(img_path)
    gray_img = Gray.(img)
    height, width = size(gray_img)
    # Create an empty matrix of Float64 to store the grayscale values
    gray_matrix = Matrix{Float64}(undef, height, width)
    # Iterate over each pixel and convert it to Float64
    for y in 1:height, x in 1:width
        gray_matrix[y, x] = float(gray_img[y, x])
    end
    return gray_matrix
end

function masking_left(x1::Vector{Float64})
    row = row1
    column = col1
    x = vectorToImage(row,column,x1)
    iter = rand(1:row)
    global row_array_left = []
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = rand(50:90)
        push!(row_array_left,row_i )
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_right(x1::Vector{Float64})
    row = row1
    column = col1
    x = vectorToImage(row,column,x1)
    iter = rand(1:row)
    global row_array_right = []
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = rand(140:180)
        push!(row_array_right, row_i)
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_r(x1::Vector{Float64})
    row = row1
    column = col1
    x = vectorToImage(row,column,x1)
    iter = length(row_array_right)
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_right[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_r(x1::Matrix{Float64})
    row = row1
    column = col1
    x2 = vec(x1)
    x = vectorToImage(row,column, x2)
    iter = length(row_array_right )
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_right[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_l(x1::Vector{Float64})
    row = row1
    column = col1
    x = vectorToImage(row,column,x1)
    iter = length(row_array_left)
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_left[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function masking_l(x1::Matrix{Float64})
    row = row1
    column = col1
    x2 = vec(x1)
    x = vectorToImage(row,column, x2)
    iter = length(row_array_left )
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    for i in 1:iter 
        row_i = row_array_left[i]
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
        end
    end
    return imageToVector(x)
end

function matrix_to_image(X::Matrix{Float64})
    gray_image = Gray.(X)
    image = Gray.(clamp.(gray_image, 0, 1))  # Ensure pixel values are between 0 and 1
    return image
end

function blur(img::Vector{Float64})
    img = vectorToImage(row1, col1, img)
    img_blurred = imfilter(img, Kernel.gaussian(3.0))
    # img_blurred = imfilter(img, Kernel.gaussian((3, 3), 0.5))
    # img_blurred = imfilter(img, Kernel.gaussian(3, 3, 0.5))

    # println(typeof(img_blurred))
    return imageToVector(img_blurred);  
end

function blur(img::Matrix{Float64})
    img = vec(img)
    img = vectorToImage(row1, col1, img)
    img_blurred = imfilter(img, Kernel.gaussian(3.0))
    # img_blurred = imfilter(img, Kernel.gaussian((3, 3), 0.5))
    # img_blurred = imfilter(img, Kernel.gaussian(3, 3, 0.5))
    # println(typeof(img_blurred))
    return imageToVector(img_blurred);  
end
