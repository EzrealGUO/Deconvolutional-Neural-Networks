import cv2
import numpy as np
import torch
from torch.autograd import Variable


from models import *
from models import vgg
from arg_parser import *
from functools import partial

def preprocess_image(cv2img, resize_im=True):
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        cv2img = cv2.resize(cv2img, (224, 224))
    im_as_arr = np.float32(cv2img)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=False)
    return im_as_var


class FeatureVisualization():
    def __init__(self, img_path, selected_layer, trained_model, model_name):
        self.img_path = img_path
        self.selected_layer = selected_layer
        self.trained_model = trained_model
        self.features = self.trained_model.features
        self.model_name = model_name

    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img, False)
        return img

    def get_features(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        x = self.process_image()
        print(x.shape)
        layer_returns = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            return_ = x
            layer_returns.append(return_)
        return layer_returns

    def get_single_feature(self):
        features = self.get_features()
        print(feature.shape for feature in features)

        feature = features[self.selected_layer][:, 0, :, :]
        print(feature.shape)

        feature = feature.view(feature.shape[1], feature.shape[2])
        print(feature.shape)

        return feature

    def save_feature_to_img(self):
        # to numpy
        feature = self.get_single_feature()
        feature = feature.data.numpy()

        # # use sigmod to [0,1]
        # feature = 1.0 / (1 + np.exp(-1 * feature))

        # absolute values after deconvolution
        feature = 1.0 / (1 + np.exp(-1 * feature)) -0.5
        feature = np.abs(feature)

        # to [0,255]
        feature = np.round(feature * 255)
        print(feature[0])

        cv2.imwrite(f'wolf_{self.model_name}_layer_{self.selected_layer}_abs.jpg', feature)


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

    model_name = 'vgg16'
    myClass = FeatureVisualization('wolf.jpg', 5, model, model_name)
    print(myClass.trained_model)

    myClass.save_feature_to_img()
