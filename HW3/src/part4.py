import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    temp = 0
    #initiate orb,bf
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        temp+= im1.shape[1]
        # TODO: 1.feature detection & matching
        kp1,des1 = orb.detectAndCompute(im1,None)
        kp2,des2 = orb.detectAndCompute(im2,None)
        matches = bf.knnMatch(des1,des2,k=2)
        goodu = []
        goodv = []
        # TODO: 2. apply RANSAC to choose best H
        for m,n in matches:
            if m.distance < 0.8 * n.distance:
                goodu.append(kp1[m.queryIdx].pt)
                goodv.append(kp2[m.trainIdx].pt)
        goodu = np.array(goodu)
        goodv = np.array(goodv)
        
        #goodu = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        #goodv = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # TODO: 3. chain the homographies
        all_matches = goodu.shape[0]
        prob_success = 0.99
        sample_points_size = 5
        ratio_of_outlier = 0.5
        iter = int(np.log(1.0 - prob_success)/np.log(1 - (1 - ratio_of_outlier)**sample_points_size))
        threshold = 4
        max_inlier = 0
        temp_best_H = np.eye(3)   
        for i in range(iter):
            rand_index = np.random.choice(all_matches, sample_points_size, replace=False)
            H = solve_homography(goodv[rand_index], goodu[rand_index])

            onerow = np.ones((1,len(goodu)))
            v = np.concatenate( (np.transpose(goodv), onerow), axis=0)
            u = np.concatenate( (np.transpose(goodu), onerow), axis=0)             
            v = np.dot(H,v)
            v[:2,:] = v[:2,:] / v[2,:][np.newaxis,:]
            err  = np.linalg.norm((v-u)[:-1,:], ord=1, axis=0)
            inlier = sum(err<threshold)
            
            
            if inlier > max_inlier:
                max_inlier = inlier
                temp_best_H = H
        # TODO: 4. apply warping
        last_best_H = last_best_H.dot(temp_best_H)

        out = warping(im2, dst, last_best_H, 0, im2.shape[0], temp, temp+im2.shape[1], direction='b') 
        
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)