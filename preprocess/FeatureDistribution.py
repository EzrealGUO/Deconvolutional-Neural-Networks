import cv2
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from models import *
from models import vgg
from arg_parser import *
from functools import partial
import seaborn as sns



transform = torchvision.transforms.ToTensor()





def plot_hist(feature1,feature2):
    plt.hist(feature1,bins=200,label="before deconv",color="green")
    plt.hist(feature2,bins=200,label="after deconv",color="blue")
    plt.legend()
    plt.title("histogram of signals")
    plt.savefig("aaa")
    plt.show()
    plt.close()

def plot_dense(feature1, feature2):
    log1=np.log10(feature1)
    log2=np.log10(feature2)
    sns.kdeplot(feature1,label="before deconv",color="green")
    sns.kdeplot(feature2,label="after deconv",color="blue")

    plt.legend()
    plt.title("density gram of signals")
    plt.savefig("bbb")
    plt.show()
    ##view the feature map
if __name__ == '__main__':

    args = parse_args()

    if args.deconv:
        args.deconv = partial(FastDeconv, bias=args.bias, eps=args.eps, n_iter=args.deconv_iter, block=args.block,
                              sampling_stride=args.stride)
    else:
        args.deconv = None

    if args.delinear:
        args.channel_deconv = None
        if args.block_fc > 0:
            args.delinear = partial(Delinear, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter)
        else:
            args.delinear = None
    else:
        args.delinear = None
        if args.block_fc > 0:
            args.channel_deconv = partial(ChannelDeconv, block=args.block_fc, eps=args.eps, n_iter=args.deconv_iter,
                                          sampling_stride=args.stride)
        else:
            args.channel_deconv = None

    model = vgg.VGG('VGG16', deconv=args.deconv, delinear=args.delinear, channel_deconv=args.channel_deconv)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='.././data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # print(images[0])
    IMAGE = images[0]
    im = IMAGE.unsqueeze(0)


    model(im)
    f1, f2 = model.get_feature_map()
    print(f1.shape, f2.shape)
    feature1 = torch.flatten(f1).detach().cpu().numpy()
    feature2 = torch.flatten(f2).detach().cpu().numpy()
    device = torch.device("cpu")
    print(feature1.shape)
    print(feature2.shape)

    plot_hist(feature1,feature2)
    plot_dense(feature1,feature2)

