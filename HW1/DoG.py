import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        width,height = image.shape
        #print(width,height)
        gaussian_images = []
        for octave in range(self.num_octaves):
            sameoctave = []
            sameoctave.append(image)
            for i in range(self.num_DoG_images_per_octave):
                blurred_img = cv2.GaussianBlur(image,ksize=(0,0),sigmaX=self.sigma**(i+1))
                sameoctave.append(blurred_img)
            gaussian_images.append(sameoctave)
            image = sameoctave[-1] #most blurred
            image = cv2.resize(image,(height//2,width//2),interpolation=cv2.INTER_NEAREST)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for sameoctave in gaussian_images:
            sub_dog = []
            for img_idx in range(len(sameoctave)-1):
                new_img = cv2.subtract(sameoctave[img_idx+1], sameoctave[img_idx])
                sub_dog.append(new_img)
            sub_dog = np.array(sub_dog)

            dog_images.append(sub_dog)

        #for report
        for oct_idx in range(self.num_octaves):
            for j in range(4):
                temp_min = np.min(dog_images[oct_idx][j])
                temp_max = np.max(dog_images[oct_idx][j])
                report_dog = (dog_images[oct_idx][j] - temp_min)/(temp_max-temp_min) * 255
                
                ''' for report DOG image
                cv2.imshow("DOG_img.jpg",report_dog)
                cv2.imwrite('../report_img/DoG_img%d_%d.png' % (oct_idx, j),report_dog)
                '''
        
        # Step 3: Thresholding the value and Find local extremum (local maximum and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        #threshold = 3
        oct = 1
        for sub in dog_images:
            for img_idx in range(1,3):
                #print(len(sub[img_idx][0]),len(sub[img_idx][1]))
                #print(sub[img_idx].shape)
                for i in range(1,sub[img_idx].shape[0]-2):
                    for j in range(1,sub[img_idx].shape[1]-2):
                        #print(i,j)
                        center_pixel = sub[img_idx][i][j]
                        if(abs(center_pixel) < self.threshold):
                            continue
                        else:##start from here
                            #print(type(sub))
                            temp = [sub[img_idx-1:img_idx+2,i-1:i+2,j-1:j+2]]
                            if(center_pixel == np.max(temp) or center_pixel == np.min(temp)):
                                if(oct == 2):
                                    keypoints.append([2*i,2*j])
                                else:
                                    keypoints.append([i,j])
                                    #print(center_pixel)
                                    #print(temp)
            oct = oct+1

                            
        keypoints = np.array(keypoints)
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
