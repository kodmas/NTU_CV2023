
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

        #create spatial kernel
        self.r = self.wndw_size//2
        self.spatial_kernel =  [[np.exp(-(i**2+j**2)/(2*self.sigma_s**2)) for j in range(-self.r,self.r+1)] for i in range(-self.r,self.r+1)]
        
        self.spatial_kernel = np.array(self.spatial_kernel)
       
        self.exponent_table = np.exp(-np.arange(256)/255* np.arange(256)/255/(2*self.sigma_r**2))
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        #height,width,channel = img.shape

        output = np.zeros(img.shape) #initialize output

        # use table to minimize exponential calculation
        
        #padded_guidance = padded_guidance/255  ## divide 255 for normalize
        #g_height,g_width,g_channel = guidance.shape
        
        ### TODO ###
        # distinguish one/three channel(s) guidance image
        #print(guidance.shape)
        if(guidance.ndim == 2):
            for i in range(self.r,self.r + img.shape[0]):
                for j in range(self.r, self.r + img.shape[1]):
                    hr = self.exponent_table[np.abs(padded_guidance[i-self.r:i+self.r+1,j-self.r:j+self.r+1] - padded_guidance[i,j])]
                    #H = np.matmul(hr,self.spatial_kernel)
                    H = np.multiply(hr,self.spatial_kernel)
                    pixel = padded_img[i-self.r:i+self.r+1,j-self.r:j+self.r+1]
                    #print(H.shape)
                    W = np.sum(H)
                    #print(W.shape)
                    output[i-self.r,j-self.r] =  np.sum(np.multiply(H[:,:,np.newaxis],pixel),axis=(0,1))/W
        elif(guidance.ndim == 3):
            for i in range(self.r,self.r + img.shape[0]):
                for j in range(self.r, self.r + img.shape[1]):
                    hr = self.exponent_table[np.abs(padded_guidance[i-self.r:i+self.r+1,j-self.r:j+self.r+1,0] - padded_guidance[i,j,0])]* \
                    self.exponent_table[np.abs(padded_guidance[i-self.r:i+self.r+1,j-self.r:j+self.r+1,1] - padded_guidance[i,j,1])]* \
                    self.exponent_table[np.abs(padded_guidance[i-self.r:i+self.r+1,j-self.r:j+self.r+1,2] - padded_guidance[i,j,2])]
                    #H = np.matmul(self.spatial_kernel,hr)
                    #H = hr*self.spatial_kernel
                    H = np.multiply(hr,self.spatial_kernel)
                    #print(H.shape)
                    pixel = padded_img[i-self.r:i+self.r+1,j-self.r:j+self.r+1]
                    #print(pixel.shape)
                    W = np.sum(H)
                    #print(W)
                    output[i-self.r,j-self.r] =  np.sum(np.multiply(H[:,:,np.newaxis],pixel),axis=(0,1))/W
        else:
            print("Guidance Channel Error")

        return np.clip(output, 0, 255).astype(np.uint8)
