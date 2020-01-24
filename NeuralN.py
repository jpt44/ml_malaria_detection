import numpy as np
import pandas as pd
import torch
from NeuralClass import Net
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.enabled=False
# device=torch.device('cpu')
print(device)

imagesFile = "D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\images"
infectedFile = "D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\infected"
savePath = "D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria"
pth = "D:\College\Masters\Fall 2019-20\ECE 687\Project\Malaria Bounding Boxes\malaria\\training.json"

trainingdf = pd.read_csv(savePath + "\\" + "trainingDataFrame.csv", delimiter=",", header=0, index_col=0)
testingdf = trainingdf.loc[965:].copy()
trainingdf.drop(trainingdf.index[965:], inplace=True)  # drop unnecessary rows
testingdf.reset_index(drop=True, inplace=True)  # reset row numbering to start from 0
print(trainingdf)
print(testingdf)

descStore = np.asarray(np.load(savePath + "\\" + "trainingdescStore.npy", "r"))
print("DescStoreLen:", len(descStore))

lbpStore =np.asarray(np.load(savePath + "\\" + "traininglbpStoreNotFullCell.npy", "r"))
print("lbpStoreLen:", len(lbpStore))

with open(savePath + "\\" + "trainingClass.txt", "r") as f:
    classStore = [int(x) for x in f.readlines()]
print("ClassStoreLen:", len(classStore))

#================PCA HERE=====================
# pca=PCA(66)
# stScale=StandardScaler().fit_transform(descStore)
# descStore=pca.fit_transform(stScale)
# pca=PCA(190)
# stScale=StandardScaler().fit_transform(lbpStore)
# lbpStore=pca.fit_transform(stScale)

# stScale=StandardScaler().fit_transform(descStore)

#================PCA END HERE=====================

#Combine data and labels
trainSet=[]
testSet=[]
for i in range(0,trainingdf.loc[965-1,"DescIndex"]):
    # trainSet.append([descStore[i],classStore[i],lbpStore[i]])
    trainSet.append([descStore[i], classStore[i]])
for i in range(trainingdf.loc[965-1,"DescIndex"],len(descStore)):
    # testSet.append([descStore[i], classStore[i],lbpStore[i]])
    testSet.append([descStore[i], classStore[i]])

pts=trainSet[0][0].shape[0]#+trainSet[0][2].shape[0]
neurNet=Net(pts=pts).to(device)
optimizer=optim.Adam(neurNet.parameters(),lr=0.001)

# Train net
# X=torch.tensor([np.append(t[0],t[2]) for t in trainSet],device=device,dtype=torch.float32).view(-1,pts)
X=torch.tensor([t[0] for t in trainSet],device=device,dtype=torch.float32).view(-1,pts)
y=torch.tensor([t[1] for t in trainSet],dtype=torch.long,device=device)

n_epochs = 8
BATCH_SIZE = 100
for epoch in range(n_epochs):
    for i in tqdm(range(0, len(X), BATCH_SIZE)):
        batch_X = X[i:i + BATCH_SIZE].view(-1, pts).to(device)
        batch_y = y[i:i + BATCH_SIZE].to(device)
        neurNet.zero_grad()  # for every samples of batch size, 0 the gradient
        outputs = neurNet(batch_X)
        loss = F.nll_loss(outputs, batch_y)
        loss.backward()
        optimizer.step()
    # print(loss)

#test the trained neural net on training data
correct=0
total=0
truePos=0
falsPos=0
falsNeg=0
classes=y.tolist()
with torch.no_grad():
    for i in tqdm(range(len(X))):
        net_out = torch.argmax(neurNet(X[i].view(-1, pts)))
        net_out=int(net_out)
        if net_out==classes[i]:
            correct+=1
        else:
            if classes[i] == 1 and net_out == 0: falsNeg+=1
            if classes[i] == 0 and net_out == 1: falsPos+=1
        total+=1
prec=100*correct/(correct+falsPos)
recall=100*correct/(correct+falsNeg)
print("\nTraining Data Accuracy:",100*correct/total)
print("\nTraining Data Recall:",recall)
print("\nTraining Data Precision:",prec)
print("\nTraining Data F-score:",2*(prec*recall/(prec+recall)))

#test the trained neural net on Testing data
# X=torch.tensor([np.append(t[0],t[2]) for t in testSet],device=device,dtype=torch.float32).view(-1,pts)
X=torch.tensor([t[0] for t in testSet],device=device,dtype=torch.float32).view(-1,pts)
y=torch.tensor([t[1] for t in testSet],dtype=torch.long,device=device)
correct=0
total=0
truePos=0
falsPos=0
falsNeg=0
classes=y.tolist()
with torch.no_grad():
    for i in tqdm(range(len(X))):
        net_out = torch.argmax(neurNet(X[i].view(-1, pts)))
        net_out=int(net_out)
        if net_out==classes[i]:
            correct+=1
        else:
            if classes[i] == 1 and net_out == 0: falsNeg+=1
            if classes[i] == 0 and net_out == 1: falsPos+=1
        total+=1
prec=100*correct/(correct+falsPos)
recall=100*correct/(correct+falsNeg)
print("\nTesting Data Accuracy:",100*correct/total)
print("\nTesting Data Recall:",recall)
print("\nTesting Data Precision:",prec)
print("\nTesting Data F-score:",2*(prec*recall/(prec+recall)))

print("# of Infected:", sum(np.where(classes==1)))
print("# of UnInfected:", sum(np.where(classes==0)))
print("# of UnInfected:", len(classes))