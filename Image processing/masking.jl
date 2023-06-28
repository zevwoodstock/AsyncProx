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

function masking(row::Int64,column::Int64,x::Matrix{Float64})
    iter = rand(1:row)
    n = length(x)
    L::Matrix{Float64} = ones(row,column)
    # println("printing iter ",iter)
    for i in 1:iter 
        row_i = rand(1:row)
        # println("printing row_i ",row_i)
        # L[row*column*(row_i-1) + row_i] = 0.0
        j = row_i
        for j in 1:column
            L[(j-1)*row + row_i] = 0.0
            x[(j-1)*row + row_i] = 0.0
            # println("PRINTING J ",j)
        end
    end
    return (x,L)
end

function matrix_to_image(X::Matrix{Float64})
    gray_image = Gray.(X)
    image = Gray.(clamp.(gray_image, 0, 1))  # Ensure pixel values are between 0 and 1
    return image
end

path1 = "/Users/kashishgoel/Downloads/image2.jpeg"
img_arr = image_to_vector(path1)
global (row, column) = get_row_column(path1)
matrix_to_image(img_arr)
y,L = masking(row,column,img_arr)
matrix_to_image(y)
# println(L)