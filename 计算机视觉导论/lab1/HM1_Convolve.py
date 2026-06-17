import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":
        p_h = img.shape[0]+2*padding_size
        p_w = img.shape[1]+2*padding_size
        padding_img = np.zeros((p_h, p_w))
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        return padding_img
    elif type=="replicatePadding":
        p_h = img.shape[0]+2*padding_size
        p_w = img.shape[1]+2*padding_size
        padding_img = np.zeros((p_h, p_w))
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        padding_img[:padding_size, padding_size:padding_size+img.shape[1]] = img[0,:]
        padding_img[-padding_size:, padding_size:padding_size+img.shape[1]] = img[-1,:]
        padding_img[:, :padding_size] = padding_img[:, padding_size:padding_size+1]
        padding_img[:, -padding_size:] = padding_img[:, -padding_size-1:-padding_size]
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, 1, "zeroPadding")
    #build the Toeplitz matrix and compute convolution
    padding_img_size = (padding_img.shape[0], padding_img.shape[1])
    toe_mat = np.zeros((img.shape[0]*img.shape[1], padding_img_size[0]*padding_img_size[1]))
    x = padding_img.flatten()

    # need to compute teo_mat@x
    # assign the value to toe_mat according to the convolution operation (no for loop is allowed)
    # f_kernel = kernel.flatten()[::-1] # (k_h*k_w, )
    f_kernel = kernel.flatten()# (k_h*k_w, ) # no need to flip the kernel
    k_h, k_w = kernel.shape
    p_h, p_w = padding_img_size
    out_h, out_w = img.shape

    # For each output position, build indices of its k_h x k_w receptive field in flattened padded image.
    row_id = np.arange(out_h)[:, None]
    col_id = np.arange(out_w)[None, :]
    top_left = (row_id * p_w + col_id).reshape(-1, 1)
    kernel_offset = (np.arange(k_h)[:, None] * p_w + np.arange(k_w)[None, :]).reshape(1, -1)
    patch_idx = top_left + kernel_offset

    toe_mat[np.arange(out_h * out_w)[:, None], patch_idx] = f_kernel
    output = (toe_mat @ x).reshape(out_h, out_w)
    
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here

    img_h, img_w = img.shape
    out_h, out_w = img_h-kernel.shape[0]+1, img_w-kernel.shape[1]+1
    N = out_h * out_w
    k_h, k_w = kernel.shape
    # m_kernel = kernel.flatten()[::-1][None,:] # (1, k_h*k_w)
    m_kernel = kernel.flatten()[None,:] # (1, k_h*k_w) # no need to flip the kernel

    # assign the value to mat according to the convolution operation (no for loop is allowed)
    row_id = np.arange(out_h)[:, None]
    col_id = np.arange(out_w)[None, :]
    top_left = (row_id * img_w + col_id).reshape(-1, 1)
    kernel_offset = (np.arange(k_h)[:, None] * img_w + np.arange(k_w)[None, :]).reshape(1, -1)
    patch_idx = top_left + kernel_offset
    mat = img.flatten()[patch_idx].T # (N, k_h*k_w)
    
    # m_kernel @ mat
    output = (m_kernel @ mat).reshape(out_h, out_w)
    
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "zeroPadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("Lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    
