import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """

    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    # quantize the gradient direction into 8 directions (0, 45, 90, 135, 180, 225, 270, 315)

    standard_grad_dir = np.arange(8) * (np.pi / 4)
    ang = (grad_dir + 2 * np.pi) % (2 * np.pi)
    diff = np.abs(ang[..., None] - standard_grad_dir[None, None, :]) # (H, W, 8)
    diff = np.minimum(diff, 2*np.pi - diff)
    quantized_idx = np.argmin(diff, axis=-1) # (H, W)
    # 0°：向右 x+0,y+1;  90°：向下
    all_neighbor = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]) # (8, 2)

    front_delta = all_neighbor[quantized_idx] # (H, W, 2)
    back_delta = -front_delta
    # compute the front and back neighbor's coordinate
    # consider the boundary condition, you can set the gradient magnitude of the out-of-boundary pixel to be 0
    

    zero_pad_grad_mag = np.zeros((grad_mag.shape[0]+2, grad_mag.shape[1]+2))
    zero_pad_grad_mag[1:-1, 1:-1] = grad_mag


    row_id = np.arange(grad_mag.shape[0])[:, None]+1
    col_id = np.arange(grad_mag.shape[1])[None, :]+1

    front_row_id = row_id + front_delta[..., 0]
    front_col_id = col_id + front_delta[..., 1]
    back_row_id = row_id + back_delta[..., 0]
    back_col_id = col_id + back_delta[..., 1]

    dir_front_mag = zero_pad_grad_mag[front_row_id, front_col_id]
    dir_back_mag = zero_pad_grad_mag[back_row_id, back_col_id]

    # perform non-maximal suppression
    NMS_output = np.where((grad_mag >= dir_front_mag) & (grad_mag >= dir_back_mag), grad_mag, 0)

    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 1.1
    high_ratio = 1.8
    # simply connect adjacent pixels if the magnitude is larger than the lower threshold
    edge_pixel_id = np.where(img > 0)
    if len(edge_pixel_id[0]) == 0:
        return np.zeros_like(img)
    edge_pixel_mag = img[edge_pixel_id]

    meanVal = edge_pixel_mag.mean()
    print(len(edge_pixel_id[0]))
    print(f"meanVal: {meanVal}")
    low_threshold = meanVal * low_ratio
    high_threshold = meanVal * high_ratio
    strong_mask = edge_pixel_mag > high_threshold

    strong_r = edge_pixel_id[0][strong_mask]
    strong_c = edge_pixel_id[1][strong_mask]
    edge_queue = list(zip(strong_r, strong_c))
    visited = np.zeros_like(img, dtype=bool)
    visited[strong_r, strong_c] = True
    while edge_queue:
        r, c = edge_queue.pop(0)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < img.shape[0]) and (0 <= nc < img.shape[1]) and (not visited[nr, nc]) and (img[nr, nc] > low_threshold):
                    visited[nr, nc] = True
                    edge_queue.append((nr, nc))
    
    output = visited.astype(float)
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    # save the magnitude
    write_img("result/HM1_Canny_magnitude.png", magnitude_grad/ magnitude_grad.max() * 255)
    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
