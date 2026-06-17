import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding




def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: array
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.

    window = np.ones((window_size,window_size))
    pad_img = padding(input_img, window_size//2, "replicatePadding")

    img_gadient_x = Sobel_filter_x(pad_img)
    img_gadient_y = Sobel_filter_y(pad_img)
    img_gradient_xx = img_gadient_x * img_gadient_x
    img_gradient_yy = img_gadient_y * img_gadient_y
    img_gradient_xy = img_gadient_x * img_gadient_y

    
    I_xx = convolve(img_gradient_xx, window)
    I_yy = convolve(img_gradient_yy, window)
    I_xy = convolve(img_gradient_xy, window)

    # no need to find the egenvalues of M
    det = I_xx * I_yy - I_xy * I_xy
    trace = I_xx + I_yy
    theta = det - alpha * trace * trace


    corner_list_idx = theta > threshold
    corner_theta = theta[corner_list_idx]
    corner_list = np.concatenate((np.argwhere(corner_list_idx), corner_theta[:, None]), axis=1) # (N, 3), each row is (index of row, index of col, theta)

    return corner_list # array, each row contains information about one corner, namely (index of row, index of col, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 30

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
