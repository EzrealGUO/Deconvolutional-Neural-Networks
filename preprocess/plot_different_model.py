import matplotlib.pyplot as plt
import numpy as np
def getLossACC(logFile):
    f=open(logFile,"r",encoding='utf-8')
    line = f.readline()  # read by line
    line = f.readline()  # skip 1st line
    lossArr = []
    AccArr=[]
    current_epoch_loss=[]
    epoch=1
    while line:

        if(line!='\n'):
            nameArr = line.split("\t")

            lossArr.append(float(nameArr[1]))
            AccArr.append(float(nameArr[3]))

        line = f.readline()

    f.close()
    return lossArr,AccArr


def plot_multi_curve(data_list,ylabel,label_list,color_list):
    num=len(data_list)
    for i in range(num):
        #x = [x + 1 for x in range(len(data_list[i]))]
        #print(data_list[i])
        plt.plot(data_list[i],label=label_list[i],color=color_list[i])
    plt.xlabel("epochs")

    plt.ylabel(ylabel)
    #plt.ylim(1)
    plt.legend(label_list)
    plt.show()





loss1,_=getLossACC("../checkpoints/fashion,mlp,ep.60,SGD,0.1,cosine,bs.64,wd.0.001,bn.0,deconv.0,delinear.True,b.64,stride.3,it.5,eps.1e-05,bias.True,bfc.0/05-13-01.18/train.log")
_,acc1=getLossACC("../checkpoints/fashion,mlp,ep.60,SGD,0.1,cosine,bs.64,wd.0.001,bn.0,deconv.0,delinear.True,b.64,stride.3,it.5,eps.1e-05,bias.True,bfc.0/05-13-01.18/test.log")
loss2,_=getLossACC("../checkpoints/fashion,mlp,ep.60,SGD,0.1,cosine,bs.64,wd.0.001,bn.0,deconv.1,delinear.True,b.64,stride.3,it.5,eps.1e-05,bias.True,bfc.512/05-12-18.28/train.log")
_,acc2=getLossACC("../checkpoints/fashion,mlp,ep.60,SGD,0.1,cosine,bs.64,wd.0.001,bn.0,deconv.1,delinear.True,b.64,stride.3,it.5,eps.1e-05,bias.True,bfc.512/05-12-18.28/test.log")
loss3,_=getLossACC("../checkpoints/fashion,mlp,ep.60,SGD,0.1,cosine,bs.64,wd.0.001,bn.1,deconv.0,delinear.True,b.64,stride.3,it.5,eps.1e-05,bias.True,bfc.0/05-13-00.54/train.log")
_,acc3=getLossACC("../checkpoints/fashion,mlp,ep.60,SGD,0.1,cosine,bs.64,wd.0.001,bn.1,deconv.0,delinear.True,b.64,stride.3,it.5,eps.1e-05,bias.True,bfc.0/05-13-00.54/test.log")


LOSS=[loss1,loss2,loss3]
ACC=[acc1,acc2,acc3]
LABEL=["SGD","deconv","SGD+BN"]
COLOR=["blue","red","orange"]


print(loss1)
print(loss2)
print(acc1)
print(acc2)
plot_multi_curve(LOSS,"loss curve",LABEL,COLOR)

plot_multi_curve(ACC,"acc curve",LABEL,COLOR)


#plot_loss([1.2,2.1,3,3.3,9])