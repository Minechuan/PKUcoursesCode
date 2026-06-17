import numpy as np
from utils import draw_save_plane_with_points, normalize



'''
Compute the N loop

1-(1-(1-e)^3)^N >= 0.999

'''
N = np.log(1-0.999)/np.log(1-(1-30/130)**3)

int_N = round(N)
if int_N < N:
    int_N += 1
print("The minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9% is: ", int_N)



if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    sample_time = int_N
    distance_threshold = 0.05

    # sample points group
    num_points = noise_points.shape[0]
    assert num_points == 130, "The number of points should be 130, including 100 inliers and 30 outliers."
    sample_indices = np.argsort(np.random.rand(sample_time, num_points), axis=1)[:, :3]

    # estimate the plane with sampled points group
    p1 = noise_points[sample_indices[:, 0]]
    p2 = noise_points[sample_indices[:, 1]]
    p3 = noise_points[sample_indices[:, 2]]

    normals = np.cross(p2 - p1, p3 - p1)
    normal_norms = np.sqrt(np.sum(normals * normals, axis=1))

    # the plane function can be formulated as: A*x+B*y+C*z+D=0, 
    # where [A,B,C] is the normal vector and D is the distance from the plane to the origin. 
    # We can calculate D with one of the sampled points, here we use p1. 
    # The distance from the plane to the origin is - (A*x+B*y+C*z)
    ds = -np.sum(normals * p1, axis=1) 
    hypotheses = np.concatenate([normals, ds[:, None]], axis=1)

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    distances = np.abs(hypotheses[:, :3] @ noise_points.T + hypotheses[:, 3:4]) / normal_norms[:, None]
    inlier_mask = distances < distance_threshold
    inlier_counts = np.sum(inlier_mask, axis=1)
    best_idx = np.argmax(inlier_counts)
    best_inliers = noise_points[inlier_mask[best_idx]]

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    if best_inliers.shape[0] < 3:
        best_inliers = noise_points

    centroid = np.mean(best_inliers, axis=0)
    centered = best_inliers - centroid # center the inliers to the centroid, which can improve the numerical stability of SVD.

    # The least-variance direction is the optimal plane normal in TLS sense.
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1] 
    # The least-variance direction is the optimal plane normal in TLS sense.
    
    # The plane must pass through the centroid.
    d = -np.dot(normal, centroid)
    pf = np.array([normal[0], normal[1], normal[2], d])
    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
