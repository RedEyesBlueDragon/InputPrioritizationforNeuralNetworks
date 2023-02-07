import torch as t
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Subset
import torch.nn.functional as F
import math
import statistics


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])



mnist_trainset = datasets.MNIST(root='./data2', train=True, download=True, transform=transform)
subset_train = Subset(mnist_trainset, indices=range(len(mnist_trainset) // 1))

train_loader = t.utils.data.DataLoader(subset_train, batch_size=8, shuffle=True)

mnist_testset = datasets.MNIST(root='./data2', train=False, download=True, transform=transform)
subset_test = Subset(mnist_testset, indices=range(len(mnist_testset) // 1))

test_loader = t.utils.data.DataLoader(subset_test, batch_size=1, shuffle=True)



print("- Training-set:\t\t{}".format(len(train_loader)))
print("- Testing-set:\t\t{}".format(len(test_loader)))




def countKNeurons(layer, lay_num):
  tmp = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0]
  
  if isTrain:
    for k in range(len(layer)):
         
      for i in range(10):
        tmp[i] = t.softmax(layer[k],0 )[i]
        #print(layer[0][i] , layer[1][i])

      
      tmpIndex = sorted(range(len(tmp)), key=lambda k: tmp[k], reverse = True)
        
      #print(top1, top2, top3, top4)
      if lay_num == 1:
        if tmp[tmpIndex[0]]!=0:
          layer1Num[tmpIndex[0]] += 1
        if tmp[tmpIndex[1]]!=0:
          layer1Num[tmpIndex[1]] += 1
        if tmp[tmpIndex[2]]!=0:
          layer1Num[tmpIndex[2]] += 1

      if lay_num == 2:
        if tmp[tmpIndex[0]]!=0:
          layer2Num[tmpIndex[0]] += 1
        if tmp[tmpIndex[1]]!=0:
          layer2Num[tmpIndex[1]] += 1
        if tmp[tmpIndex[2]]!=0:
          layer2Num[tmpIndex[2]] += 1

      if lay_num == 3:
        if tmp[tmpIndex[0]]!=0:
          layer3Num[tmpIndex[0]] += 1
        if tmp[tmpIndex[1]]!=0:
          layer3Num[tmpIndex[1]] += 1
        if tmp[tmpIndex[2]]!=0:
          layer3Num[tmpIndex[2]] += 1    
    #print(layer1Num)

  else:  
    for i in range(10):
      tmp[i] = t.softmax(layer[0],0 )[i]
    #print(tmp)  


    tmpIndex = sorted(range(len(tmp)), key=lambda k: tmp[k], reverse = True)

    #print(tmp)    
    #print(top1, top2, top3, top4)
    if lay_num == 1:
      score.append(tmp[l1[0]] + tmp[l1[1]]  + tmp[l1[2]])
      score2.append(tmp[l1[0]] + tmp[l1[1]] )
    if lay_num == 2:
      score.append(tmp[l2[0]] + tmp[l2[1]] + tmp[l2[2]])
      score2.append(tmp[l2[0]]  + tmp[l2[1]]  )
    if lay_num == 3:
      score.append(tmp[l3[0]] + tmp[l3[1]] + tmp[l3[2]])
      score2.append(tmp[l3[0]]  + tmp[l3[1]]  )

    #print(layer1Num)




class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.linear1 = nn.Linear(28*28, 10) 
    self.linear2 = nn.Linear(10, 10) 
    self.linear3 = nn.Linear(10, 10)

  def forward(self, img): #convert + flatten
    #x = img.view(-1, 28*28)
    x = img
    #print(img.shape)
    out = {}
    x = self.linear1(x)
    x = F.relu(x)
    out['layer1'] = x.detach()
    countKNeurons(out['layer1'], 1)
    #print("layer1- ",out['layer1'])

    x = self.linear2(x)
    x = F.relu(x)
    out['layer2'] = x.detach()
    countKNeurons(out['layer2'], 2)
    #print("layer2- ",out['layer2'])

    x = self.linear3(x)
    out['layer3'] = x.detach()
    countKNeurons(out['layer3'], 3)
    #print("layer3- ",out['layer3'])
    return x

net = Net()




cross_el = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net.parameters(), lr=0.001) #e-1
epochs = 5


def train():
  for epoch in range(epochs):
    print(epoch)
    net.train()

    for data in train_loader:
      x, y = data
      optimizer.zero_grad()
      #print("---", x.shape)
      output = net(x.view(-1, 28*28))
      loss = cross_el(output, y)
      #print(loss)
      loss.backward()
      optimizer.step()


def test():
  correct, total = 0,0
  index = 0
  with t.no_grad():
    for data in test_loader:
      x, y = data
      #print("---", x.shape)
      output = net(x.view(-1, 28*28))

      
      for idx, i in enumerate(output):
        print(t.argmax(i) == y[idx], "--", score2[index] + score2[index+1] + score2[index+2])
        sc = t.softmax(i, 0)
        normalLst = []
        scoreSoftmax = 0
        scoreStd = 0
        scoreSoftAvg = 0

        for j in sc:
          normalLst.append(j.item())
        scoreStd = statistics.stdev(normalLst)

        for j in range(len(sc)):
          scoreSoftmax += (sc[j] * math.log(sc[j]))

        for j in range(len(sc)):
          scoreSoftAvg += sc[j] * abs(sc[j] - (sum(sc)/len(sc)))

        if t.argmax(i) == y[idx]:
          res.append([1,  scoreSoftmax, scoreStd, scoreSoftAvg, score[index] + score[index+1] + score[index+2], 1, score2[index] + score2[index+1] + score2[index+2]])
        else: 
          res.append([0,  scoreSoftmax, scoreStd, scoreSoftAvg, score[index] + score[index+1] + score[index+2], 0, score2[index] + score2[index+1] + score2[index+2]])  
        index+=3

        if t.argmax(i) == y[idx]:
          correct +=1
        total +=1
      
    print(f'accuracy: % {round(correct/total, 3) * 100}')
      


layer1Num = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0]
layer2Num = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0]
layer3Num = [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0]




score = []
score2 = []

res = []

isTrain = True
train()

print(layer1Num)
l1 = sorted(range(len(layer1Num)), key=lambda k: layer1Num[k], reverse=True)
print(l1)

l2 = sorted(range(len(layer2Num)), key=lambda k: layer2Num[k], reverse=True)
l3 = sorted(range(len(layer3Num)), key=lambda k: layer3Num[k], reverse=True)

isTrain = False
#print("----testing----")
test()

def addPlt(res, colour):
  graph = []
  coun = 0
  for x in res:
    if x[0] == 0:
      coun+=1
    graph.append(coun)  

  plt.plot(range(len(graph)), graph, color = colour)

graph1 = []
coun = 0
for x in res:
  if x[0] == 0:
    coun+=1
  graph1.append(coun)  
plt.plot(range(len(graph1)), graph1, color="red")
#plt.show()

res.sort(key=lambda x: x[1])
#addPlt(res, "blue")

res.sort(key=lambda x: x[2])
#addPlt(res, "yellow")

res.sort(key=lambda x: x[3])
#addPlt(res, "green")

res.sort(key=lambda x: x[4], reverse=True)
addPlt(res, "grey")

res.sort(key=lambda x: x[5])
addPlt(res, "black")

res.sort(key=lambda x: x[6], reverse=True)
addPlt(res, "magenta")
#print(res)

plt.xlabel("Number of Test Cases")
plt.ylabel("Number of Failed Test Cases")
plt.show()
#print(graph)

#tst = [[0.0000, 1.0614, 0.5104, 0.0000, 1.1646, 0.0000, 0.0000, 0.7842, 0.0000,0.0000],[0.0000, 0.6202, 0.6631, 0.0000, 1.4387, 0.0000, 0.0000, 1.4520, 0.7021, 0.0000]]
#countKNeurons(tst)




















