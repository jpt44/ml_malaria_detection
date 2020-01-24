import cv2
from skimage.feature import local_binary_pattern

import numpy as np

class LocalBinaryPatterns:

    def __init__(self,numPoints,radius):
        self.numPoints=numPoints
        self.radius=radius

    # def describe(self, image,eps=1e-7):
    #     # compute the Local Binary Pattern representation
    #     # of the image, and then use the LBP representation
    #     # to build the histogram of patterns
    #     lbp=local_binary_pattern(image,self.numPoints,
    #                              self.radius,method="uniform")
    #     (hist,_)=np.histogram(lbp.ravel(),
    #                         bins=np.arange(0,self.numPoints+3),
    #                         range=(0,self.numPoints+2))
    #     hist=hist.astype(np.float)
    #     hist/=(hist.sum()+eps)
    #     print(lbp)
    #     print(hist.size)
    #     cv2.imshow(None,lbp)
    #     cv2.waitKey(0)
    #     exit()
    #     return hist

    def lbpCalc(self,image):
        gray_image=image
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgLBP = np.zeros_like(gray_image)
        neighboor = 3
        for ih in range(0, image.shape[0] - neighboor):
            for iw in range(0, image.shape[1] - neighboor):
                ### Step 1: 3 by 3 pixel
                img = gray_image[ih:ih + neighboor, iw:iw + neighboor]
                center = img[1, 1]
                img01 = (img >= center) * 1.0
                # img01_vector = img01.T.flatten()
                # it is ok to order counterclock manner
                img01_vector = img01.flatten()
                ### Step 2: **Binary operation**:
                img01_vector = np.delete(img01_vector, 4)
                ### Step 3: Decimal: Convert the binary operated values to a digit.
                where_img01_vector = np.where(img01_vector)[0]
                if len(where_img01_vector) >= 1:
                    num = np.sum(2 ** where_img01_vector)
                else:
                    num = 0
                imgLBP[ih + 1, iw + 1] = num
        return imgLBP


    def describe(self, image,eps=1e-7):
        lbp=local_binary_pattern(image,self.numPoints,
                                 self.radius,method="uniform")
        hist,_=np.histogram(lbp.ravel(),
                            bins=256,
                            range=[0,256])
        hist=hist.astype(np.float)
        hist=hist/len(hist)
        print(image[0])
        print(lbp[0])
        print(hist.size)
        cv2.imshow(None,lbp)
        cv2.waitKey(0)
        return hist

