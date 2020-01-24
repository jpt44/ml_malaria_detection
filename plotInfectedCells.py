import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from LBP import LocalBinaryPatterns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os


# Plot gametocytes, rings, schizonts,trophozites look for separation
infectedCellPath="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\infected"
savePath="D:\College\Masters\Fall 2019-20\ECE 687\Project"
imagesPath="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\images"
pptImages="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\pptImages"

#Create Histograms of SIFT
# sif=cv2.xfeatures2d_SIFT.create()
# n=np.arange(0,128)
# gametoCount=schizontCount=ringCount=trophoCount=0
# gHist=tHist=scHist=ringHist=None
# for file in os.listdir(infectedCellPath):
#     print(file)
#     img = cv2.imread(infectedCellPath + "\\" + file, cv2.IMREAD_GRAYSCALE)
#     kp, desc = sif.detectAndCompute(img, None)
#     hist,bin_edges=np.histogram(desc,bins=128,range=[0,128])
#     hist=hist/hist.shape[0]
#     if file.__contains__("gameto"):
#         if gametoCount==0:
#             gHist=hist.copy()
#         else:
#             gHist=gHist + hist
#         gametoCount+=1
#     if file.__contains__("schiz"):
#         if schizontCount == 0:
#             scHist = hist.copy()
#         else:
#             scHist = scHist + hist
#         schizontCount += 1
#     if file.__contains__("ring"):
#         if ringCount == 0:
#             ringHist = hist.copy()
#         else:
#             ringHist = ringHist + hist
#         ringCount += 1
#     if file.__contains__("trophozoite"):
#         if trophoCount == 0:
#             tHist = hist.copy()
#         else:
#             tHist = tHist + hist
#         trophoCount += 1
#
# gHist=gHist/gametoCount
# tHist=tHist/trophoCount
# scHist=scHist/schizontCount
# ringHist=ringHist/ringCount
# fontS=15
# fig, axs = plt.subplots(2, 2)
# axs[0,0].hist(x=n,bins=128,weights=gHist,rwidth=0.85)
# axs[0,0].set_ylabel('Normalized Count',fontsize=fontS)
# axs[0,0].set_xlabel('Bin',fontsize=fontS)
# axs[0,0].set_xlim([-1,129])
# axs[0,0].grid(True)
# axs[0,0].set_title("Gametocyte SIFT Descriptors",fontsize=fontS)
# axs[0,0].set_axisbelow(True)
#
# axs[0,1].hist(x=n,bins=128,weights=ringHist,rwidth=0.85)
# axs[0,1].set_ylabel('Normalized Count',fontsize=fontS)
# axs[0,1].set_xlabel('Bin',fontsize=fontS)
# axs[0,1].set_xlim([-1,129])
# axs[0,1].grid(True)
# axs[0,1].set_title("Ring SIFT Descriptors",fontsize=fontS)
# axs[0,1].set_axisbelow(True)
#
# axs[1,0].hist(x=n,bins=128,weights=scHist,rwidth=0.85)
# axs[1,0].set_ylabel('Normalized Count',fontsize=fontS)
# axs[1,0].set_xlabel('Bin',fontsize=fontS)
# axs[1,0].set_xlim([-1,129])
# axs[1,0].grid(True)
# axs[1,0].set_title("Schizont SIFT Descriptors",fontsize=fontS)
# axs[1,0].set_axisbelow(True)
#
# axs[1,1].hist(x=n,bins=128,weights=tHist,rwidth=0.85)
# axs[1,1].set_ylabel('Normalized Count',fontsize=fontS)
# axs[1,1].set_xlabel('Bin',fontsize=fontS)
# axs[1,1].set_xlim([-1,129])
# axs[1,1].grid(True)
# axs[1,1].set_title("Trophozoite SIFT Descriptors",fontsize=fontS)
# axs[1,1].set_axisbelow(True)
#
# fig.suptitle('SIFT Calculated Descriptors',fontsize=fontS)
# plt.show()
# exit()
#=================PCA ON SIFT========================
sif=cv2.xfeatures2d_SIFT.create()
lb=LocalBinaryPatterns()
n=np.arange(0,128)
gametoCount=schizontCount=ringCount=trophoCount=0
gHist=tHist=scHist=ringHist=None
pca=PCA(64)
# descStore=pca.fit_transform(stScale)

for file in os.listdir(infectedCellPath):
    print(file)
    img = cv2.imread(infectedCellPath + "\\" + file, cv2.IMREAD_GRAYSCALE)
    kp, desc = sif.detectAndCompute(img, None)
    stScale = StandardScaler().fit_transform(desc)
    desc = pca.fit_transform(stScale)
    # lpImg = lb.lbpCalc(img)
    # t=np.append(desc,lpImg)
    # stScale = StandardScaler().fit_transform(t)
    # hist,bin_edges=np.histogram(np.append(desc,lpImg),bins=128,range=[0,128])
    hist, bin_edges = np.histogram(desc, bins=128, range=[0, 128])
    hist=hist/hist.shape[0]
    if file.__contains__("gameto"):
        if gametoCount==0:
            gHist=hist.copy()
        else:
            gHist=gHist + hist
        gametoCount+=1
    if file.__contains__("schiz"):
        if schizontCount == 0:
            scHist = hist.copy()
        else:
            scHist = scHist + hist
        schizontCount += 1
    if file.__contains__("ring"):
        if ringCount == 0:
            ringHist = hist.copy()
        else:
            ringHist = ringHist + hist
        ringCount += 1
    if file.__contains__("trophozoite"):
        if trophoCount == 0:
            tHist = hist.copy()
        else:
            tHist = tHist + hist
        trophoCount += 1

gHist=gHist/gametoCount
tHist=tHist/trophoCount
scHist=scHist/schizontCount
ringHist=ringHist/ringCount
fig, axs = plt.subplots(2, 2)
axs[0,0].hist(x=n,bins=128,weights=gHist,rwidth=0.85)
axs[0,0].set_ylabel('Normalized Count')
axs[0,0].set_xlabel('Normalized Count')
axs[0,0].set_xlim([-1,129])
axs[0,0].grid(True)
axs[0,0].set_title("Gametocyte SIFT Descriptors")
axs[0,0].set_axisbelow(True)

axs[0,1].hist(x=n,bins=128,weights=ringHist,rwidth=0.85)
axs[0,1].set_ylabel('Normalized Count')
axs[0,1].set_xlim([-1,129])
axs[0,1].grid(True)
axs[0,1].set_title("Ring SIFT Descriptors")
axs[0,1].set_axisbelow(True)

axs[1,0].hist(x=n,bins=128,weights=scHist,rwidth=0.85)
axs[1,0].set_ylabel('Normalized Count')
axs[1,0].set_xlim([-1,129])
axs[1,0].grid(True)
axs[1,0].set_title("Schizont SIFT Descriptors")
axs[1,0].set_axisbelow(True)

axs[1,1].hist(x=n,bins=128,weights=tHist,rwidth=0.85)
axs[1,1].set_ylabel('Normalized Count')
axs[1,1].set_xlim([-1,129])
axs[1,1].grid(True)
axs[1,1].set_title("Trophozoite SIFT Descriptors")
axs[1,1].set_axisbelow(True)

fig.suptitle('SIFT Calculated Descriptors')
plt.show()
exit()

#===========================RGB,GrayScale=================
img1=img2=img3=img4=None
x,x2,x3,x4=[],[],[],[]
y,y2,y3,y4=[],[],[],[]
z,z2,z3,z4=[],[],[],[]
pts=2
msk=cv2.IMREAD_GRAYSCALE
sif=cv2.xfeatures2d_SIFT.create()
lb=LocalBinaryPatterns(24,5)
for file in os.listdir(infectedCellPath):
    img=cv2.imread(imagesPath + "\\" + file,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(imagesPath + "\\" + file, cv2.IMREAD_GRAYSCALE)
    kp,desc=sif.detectAndCompute(img,None)
    kp2, desc2 = sif.detectAndCompute(img2, None)
    img3=cv2.drawKeypoints(img,kp,None,color=(0,0,0))
    img4 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    img5=cv2.imread(imagesPath + "\\" + file,cv2.IMREAD_GRAYSCALE)
    lpImg=lb.lbpCalc(img5)
    cv2.imshow(file,img3)
    cv2.imshow(None,img4)
    cv2.imshow("LBP",lpImg)
    cv2.waitKey(0)
    exit()
    r = np.random.randint(0, 50, (pts, 1))
    # # print("random",r)
    if file.__contains__("gametocyte"):
        img1=cv2.imread(infectedCellPath + "\\" + file,msk)
        bgr = img1.reshape(img1.shape[0] * img1.shape[1], 3)  # gameto
        bgr=np.reshape(bgr[r],(pts,3))
        x.extend(bgr[:,0])
        y.extend(bgr[:,1])
        z.extend(bgr[:,2])
    elif file.__contains__("schizont"):
        img2 = cv2.imread(infectedCellPath + "\\" + file,msk)
        bgr = img2.reshape(img2.shape[0] * img2.shape[1], 3)  # gameto
        bgr=np.reshape(bgr[r],(pts,3))
        x2.extend(bgr[:,0])
        y2.extend(bgr[:,1])
        z2.extend(bgr[:,2])
    elif file.__contains__("ring"):
        img3 = cv2.imread(infectedCellPath + "\\" + file,msk)
        bgr = img3.reshape(img3.shape[0] * img3.shape[1], 3)  # gameto
        bgr=np.reshape(bgr[r],(pts,3))
        x3.extend(bgr[:,0])
        y3.extend(bgr[:,1])
        z3.extend(bgr[:,2])
    elif file.__contains__("tropho"):
        img4 = cv2.imread(infectedCellPath + "\\" + file,msk)
        bgr = img4.reshape(img4.shape[0] * img4.shape[1], 3)  # gameto
        bgr=np.reshape(bgr[r],(pts,3))
        x4.extend(bgr[:,0])
        y4.extend(bgr[:,1])
        z4.extend(bgr[:,2])

# bgr=img1.reshape(img1.shape[0]*img1.shape[1],3) #gameto
# bgr2=img2.reshape(img2.shape[0]*img2.shape[1],3)
# bgr3=img3.reshape(img3.shape[0]*img3.shape[1],3)
# bgr4=img4.reshape(img4.shape[0]*img4.shape[1],3)
# start=0
# end=800
# x,y,z=bgr[start:end,0],bgr[start:end,1],bgr[start:end,2]
# x2,y2,z2=bgr2[start:end,0],bgr2[start:end,1],bgr2[start:end,2]
# x3,y3,z3=bgr3[start:end,0],bgr3[start:end,1],bgr3[start:end,2]
# x4,y4,z4=bgr4[start:end,0],bgr4[start:end,1],bgr4[start:end,2]
print(len(x))

# fig=plt.figure()
# ax=fig.add_subplot(111,projection="3d")
# ax.scatter(x,y,z,c='r',marker="o")
# ax.scatter(x2,y2,z2,c='g',marker="o")
# ax.scatter(x3,y3,z3,c='b',marker="o")
# ax.scatter(x4,y4,z4,c='k',marker="o")
# ax.set_xlabel('B',fontsize=20)
# ax.set_ylabel('G',fontsize=20)
# ax.set_zlabel('R',fontsize=20)
# ax.legend(['Gametocyte','Schizont','Ring','Trophozoite'],fontsize=20)
# plt.title('Parasite Cells Red, Green, Blue (RGB)',fontsize=20)
# plt.show()

# 2D Plots
# fig, axs = plt.subplots(3, 1)
#
# f=14

# axs[0].scatter(x,y,c='r',marker="o")
# axs[0].scatter(x2,y2,c='g',marker="o")
# axs[0].scatter(x3,y3,c='b',marker="o")
# axs[0].scatter(x4,y4,c='k',marker="o")
# # axs[0].set_title('Parasite Cells BG')
# axs[0].set_xlabel('B',fontsize=f)
# axs[0].set_ylabel('G',fontsize=f)
# axs[0].legend(['Gametocyte','Schizont','Ring','Trophozoite'],fontsize=f)
# axs[0].grid(True)
# axs[0].set_axisbelow(True)
#
# axs[1].scatter(x,z,c='r',marker="o")
# axs[1].scatter(x2,z2,c='g',marker="o")
# axs[1].scatter(x3,z3,c='b',marker="o")
# axs[1].scatter(x4,z4,c='k',marker="o")
# # axs[1].set_title('Parasite Cells BR')
# axs[1].set_xlabel('B',fontsize=f)
# axs[1].set_ylabel('R',fontsize=f)
# axs[1].legend(['Gametocyte','Schizont','Ring','Trophozoite'],fontsize=f)
# axs[1].grid(True)
# axs[1].set_axisbelow(True)
#
# axs[2].scatter(y,z,c='r',marker="o")
# axs[2].scatter(y2,z2,c='g',marker="o")
# axs[2].scatter(y3,z3,c='b',marker="o")
# axs[2].scatter(y4,z4,c='k',marker="o")
# # axs[2].set_title('Parasite Cells GR')
# axs[2].set_xlabel('G',fontsize=f)
# axs[2].set_ylabel('R',fontsize=f)
# axs[2].legend(['Gametocyte','Schizont','Ring','Trophozoite'],fontsize=f)
# axs[2].grid(True)
# axs[2].set_axisbelow(True)
#
# fig.suptitle('Parasitized and Parasite Cells Red, Blue, Green (RBG)',fontsize=f)


# fig=plt.figure()
# ax=fig.add_subplot(111)
# ax.scatter(x,y,c='r',marker="o")
# ax.scatter(x2,y2,c='g',marker="o")
# ax.scatter(x3,y3,c='b',marker="o")
# ax.scatter(x4,y4,c='k',marker="o")
# ax.set_xlabel('B')
# ax.set_ylabel('G')
# ax.legend(['Gametocyte','Schizont','Ring','Trophozoite'])
# plt.title('Parasite Cells BG')

# ax=fig.add_subplot(112)
# ax.scatter(x,y,c='r',marker="o")
# ax.scatter(x2,y2,c='g',marker="o")
# ax.scatter(x3,y3,c='b',marker="o")
# ax.scatter(x4,y4,c='k',marker="o")
# ax.set_xlabel('B')
# ax.set_ylabel('G')
# ax.legend(['Gametocyte','Schizont','Ring','Trophozoite'])
# plt.title('Parasite Cells BG')
# plt.show()

#=============================SIFT Algo=======================
# sif=cv2.xfeatures2d_SIFT.create()
# ct=0
# for file in os.listdir(imagesPath):
#     ct+=1
#     if ct>0:
#         img1=cv2.imread(imagesPath + "\\" + file)
#         kp,desc=sif.detectAndCompute(img1,None)
#         img2=cv2.drawKeypoints(img1,kp,None,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         # cv2.imshow(None,img2)
#         cv2.imwrite(pptImages + "\\sift" + str(ct) + ".png",img2)
#         # cv2.waitKey(0)
#         break
