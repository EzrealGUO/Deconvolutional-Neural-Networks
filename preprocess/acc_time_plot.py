import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import numpy

# input_1 = "vgg11_deconv_1_test_acc.csv"
# input_2 = "vgg11_deconv_0_test_acc.csv"
input_1 = "vgg11_cifar10_deconv_1_test_acc.csv"
input_2 = "vgg11_cifar10_deconv_0_test_acc.csv"

with_dc = pd.read_csv(input_1)
without_dc = pd.read_csv(input_2)

with_dc = np.array(with_dc)
without_dc = np.array(without_dc)
# print(with_dc)
# plt = plt.subplot(1, 1, 1)
ax = plt.gca()
plt.plot(with_dc[:, 1], with_dc[:, 0], c='orange', label='Deconv Top-1 Acc')
plt.plot(without_dc[:, 1], without_dc[:, 0], c='r', label='SGD + BN Top-1 Acc')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.title('VGG-11 Top-1 Validation Accuracy on CIFAR-10', pad=20)
plt.xlabel('Walltime (Hrs)', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.legend(loc=4, prop={"size": 10})
plt.savefig('acc_time_plot_vgg11_cifar10.png')
plt.show()

