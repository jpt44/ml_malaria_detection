import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import cv2
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from LBP import LocalBinaryPatterns


def renameRow(row):
    return row["pathname"].split("/")[2]

if __name__=="__main__":
    pts=128+256
    sift=cv2.xfeatures2d_SIFT.create()
    localbp=LocalBinaryPatterns(pts,4)

    imagesFile="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\images"
    infectedFile="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\infected"
    savePath="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria"
    pth="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\\training.json"
    #
    trainingdf=pd.read_json(pth)
    trainingdf["Class"]=pd.Series([0]*trainingdf.shape[0])
    trainingdf["SVM Class"]=pd.Series([np.NaN]*trainingdf.shape[0])
    trainingdf["DescIndex"]=pd.Series([0]*trainingdf.shape[0])
    trainingdf["image"]=trainingdf["image"].apply(renameRow)  #get image name and make it the value in imageColumn

    classStore=[] #class storage, infected vs uninfected (0 vs 1)
    descStore=[] #descriptors storage, stores infected and uninfected descriptors
    lbpStore=[] #local binary pattern storage, stores infected and uninfected

    desCounter=0
    for f in range(0,trainingdf.shape[0]):
        file=trainingdf.loc[f,"image"]
        img1=cv2.imread(imagesFile + "\\" + file,cv2.IMREAD_COLOR)
        kp, desc = sift.detectAndCompute(img1,None)
        print(f,": ",file)
        for k in range(len(kp)):
            for j in range(len(trainingdf["objects"].iloc[f])): #iterate through bounding boxes
                min_r=trainingdf["objects"].iloc[f][j]["bounding_box"]["minimum"]['r']
                max_r=trainingdf["objects"].iloc[f][j]["bounding_box"]["maximum"]['r']
                min_c=trainingdf["objects"].iloc[f][j]["bounding_box"]["minimum"]['c']
                max_c=trainingdf["objects"].iloc[f][j]["bounding_box"]["maximum"]['c']
                categ=trainingdf["objects"].iloc[f][j]["category"]

                #if kp is within bounding box parameters
                if min_r<=kp[k].pt[0]<=max_r and min_c<=kp[k].pt[1]<=max_c:
                    # h = localbp.describe(img1[slice(min_r, max_r), slice(min_c, max_c)])
                    center=[int(kp[k].pt[0])-8,int(kp[k].pt[0])+8,
                            int(kp[k].pt[1])-8,int(kp[k].pt[1])+8]

                    if center[0] < 0:
                        center[0],center[1] = 0,16
                    if center[2] < 0:
                        center[2],center[3] = 0,16
                    if center[1] > img1.shape[0]:
                        center[0],center[1] = img1.shape[0]-16,img1.shape[0]
                    if center[3] > img1.shape[1]:
                        center[2],center[3] = img1.shape[1]-16,img1.shape[1]

                    res=img1[slice(center[0], center[1]), slice(center[2], center[3])]
                    h = localbp.lbpCalc(res)
                    if h.flatten().shape[0]!=256:
                        print(h.flatten().shape)
                        print(center)
                    lbpStore.append(h.flatten())
                    descStore.append(desc[k])
                    desCounter+=1
                    trainingdf.loc[f,"DescIndex"]=desCounter
                    if categ in ["red blood cell","leukocyte"]:
                        classStore.append(0) #uninfected
                    else:
                        classStore.append(1)  #infected
                        trainingdf.loc[f,"Class"]=1
                    break #no need to go through rest of boxes because location of kp found


    # np.save(savePath + "\\" + 'traininglbpStore.npy',np.asarray(lbpStore,dtype=np.float))
    # np.save(savePath + "\\" + 'traininglbpStoreNotFullCell.npy', np.asarray(lbpStore,dtype=np.float))
    # np.save(savePath + "\\" + "trainingdescStore.npy",np.asarray(descStore))
    # pd.DataFrame.to_csv(trainingdf,savePath + "\\" + "trainingDataFrame.csv",columns=trainingdf.columns)
    # with open(savePath + "\\" + "trainingClass.txt","w") as f:
    #     f.writelines('%d\n' %i for i in classStore)
    # exit()

    trainingdf=pd.read_csv(savePath + "\\" + "trainingDataFrame.csv",delimiter=",",header=0,index_col=0)
    testingdf=trainingdf.loc[965:].copy()
    trainingdf.drop(trainingdf.index[965:],inplace=True) #drop unnecessary rows
    testingdf.reset_index(drop=True,inplace=True) #reset row numbering to start from 0
    print(trainingdf)
    print(testingdf)

    descStore=np.load(savePath + "\\" + "trainingdescStore.npy","r")
    descStore=[np.array(x,"float64") for x in descStore]
    print("DescStoreLen:",len(descStore))

    lbpStore=np.load(savePath + "\\" + "traininglbpStoreNotFullCell.npy","r")
    lbpStore=[np.array(x,"float64") for x in lbpStore]
    print("lbpStoreLen:",len(lbpStore))

    with open(savePath + "\\" + "trainingClass.txt","r") as f:
        classStore=[int(x.rstrip()) for x in f.readlines()]
    print("ClassStoreLen:",len(classStore))

    # t=[]
    # for i in range(len(descStore)):
    #     t.append(np.append(descStore[i],lbpStore[i]))
    # stScale = StandardScaler().fit_transform(t)
    stScale=StandardScaler().fit_transform(descStore)

    pca = PCA(66)
    # t = pca.fit_transform(stScale)
    descStore = pca.fit_transform(stScale)

    #=================Build + Train Decision Tree=====================================
    # clf=svm.SVC(gamma='auto',kernel='linear',tol=0.001,max_iter=10000) #classifier
    print("="*10+"Training SVM Classifier"+"="*10)
    #
    # n_estimators=5
    # clf=OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear', probability=True,max_iter=300), max_samples=n_estimators, n_estimators=n_estimators))
    #
    clf=RandomForestClassifier(min_samples_leaf=1,n_estimators=5,n_jobs=3,random_state=0,
                               bootstrap=True,criterion='gini')
    clf.fit(descStore[:trainingdf.loc[965 - 1, "DescIndex"]], classStore[:trainingdf.loc[965 - 1, "DescIndex"]])
    print("Training score:",100*clf.score(descStore[:trainingdf.loc[965-1,"DescIndex"]],
                                          classStore[:trainingdf.loc[965-1,"DescIndex"]]))

    # clf.fit(t[:trainingdf.loc[965 - 1, "DescIndex"]],
    #         classStore[:trainingdf.loc[965 - 1, "DescIndex"]])
    # print("Training score:",100*clf.score(t[:trainingdf.loc[965-1,"DescIndex"]],
    #                                       classStore[:trainingdf.loc[965-1,"DescIndex"]]))

    #===============Test with training data======================
    print("="*10+"Testing SVM with Training Data"+"="*10)
    for i in range(trainingdf.shape[0]):
        print(i, ": ", trainingdf.loc[i,"image"])
        # img_test=cv2.imread(imagesFile + "\\" + trainingdf.loc[i,"image"],cv2.IMREAD_GRAYSCALE)
        # kptest, destest = sift.detectAndCompute(img_test, None)
        if i==0:
            destest=descStore[:trainingdf.loc[i,"DescIndex"]]
            # destest = t[:trainingdf.loc[i, "DescIndex"]]
        else:
            destest = descStore[trainingdf.loc[i-1, "DescIndex"]:trainingdf.loc[i, "DescIndex"]]
            # destest = t[trainingdf.loc[i - 1, "DescIndex"]:trainingdf.loc[i, "DescIndex"]]

        pred = clf.predict(destest)
        trainingdf.loc[i,"SVM Class"]= 1 if 1 in pred else 0


    # print(trainingdf)
    # print(trainingdf[trainingdf["Class"]==trainingdf["SVM Class"]])
    print("Error with Training Data",
          100*trainingdf[trainingdf["Class"]!=trainingdf["SVM Class"]].shape[0]/trainingdf.shape[0])

    # #=========Test with test data====================
    # exit()
    # pth="D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\\test.json"
    #
    # testingdf=pd.read_json(pth)
    # testingdf["Class"]=pd.Series([0]*testingdf.shape[0]) #classification of image
    # testingdf["SVM Class"]=pd.Series([np.NaN]*testingdf.shape[0])
    # testingdf["image"]=testingdf["image"].apply(renameRow)  #get image name and make it the value in imageColumn

    #Iterate through testing json and classify images with infected cells as infected
    # for i in range(testingdf.shape[0]):
    #     for j in range(len(testingdf["objects"].iloc[i])):  # iterate through bounding boxes
    #         categ = testingdf["objects"].iloc[i][j]["category"]
    #         if categ not in ["red blood cell","leukocyte"]:
    #             testingdf.loc[i,"Class"]=1 #infected
    #             break

    print("="*10+"Testing SVM with Testing Data"+"="*10)
    for i in range(testingdf.shape[0]):
        print(i, ": ", testingdf.loc[i,"image"])
        # img_test=cv2.imread(imagesFile + "\\" + testingdf.loc[i,"image"],cv2.IMREAD_GRAYSCALE)
        # kptest, destest = sift.detectAndCompute(img_test, None)
        if i==0:
            destest=descStore[trainingdf.loc[965-1,"DescIndex"]:testingdf.loc[i,"DescIndex"]]
            # destest = t[trainingdf.loc[965 - 1, "DescIndex"]:testingdf.loc[i, "DescIndex"]]
        else:
            destest=descStore[testingdf.loc[i-1,"DescIndex"]:testingdf.loc[i,"DescIndex"]]
            # destest = t[testingdf.loc[i-1,"DescIndex"]:testingdf.loc[i,"DescIndex"]]
        pred = clf.predict(destest)
        testingdf.loc[i, "SVM Class"]=1 if 1 in pred else 0 #infected


    print("Error with Testing Data",
          100 * testingdf[testingdf["Class"] != testingdf["SVM Class"]].shape[0] / testingdf.shape[0])

    truePos=testingdf[(testingdf["Class"]==1) & (testingdf["SVM Class"]==1)].shape[0]
    trueNeg=testingdf[(testingdf["Class"]==0) & (testingdf["SVM Class"]==0)].shape[0]
    falsPos=testingdf[(testingdf["Class"]==0) & (testingdf["SVM Class"]==1)].shape[0]
    falsNeg=testingdf[(testingdf["Class"]==1) & (testingdf["SVM Class"]==0)].shape[0]

    prec=truePos/(truePos+falsPos)
    recall=truePos/(truePos+falsNeg)
    print("Precision",100*prec)
    print("Recall", 100 * recall)
    print("F-Score",2*(prec*recall/(prec+recall)))

    pd.DataFrame.to_csv(testingdf, savePath + "\\" + "testResults.csv", columns=testingdf.columns)

    #==================Visualize decision tree===============
    # estimators=clf.estimators_[0]
    # print("Creating Graph")
    # dot_data=StringIO()
    # export_graphviz(estimators,out_file=dot_data,class_names=['0','1'],rounded=True,
    #                 impurity=False,
    #                 proportion=False,precision=2,filled=True,special_characters=True)
    #
    # print("here3")
    # graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    # print("here")
    # graph.write_png(savePath + "\\tree.png")