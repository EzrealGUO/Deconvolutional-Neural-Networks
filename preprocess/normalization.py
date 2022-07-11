import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img_path = "wolf.jpg"

# open Image
img = Image.open(img_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# transformation: [H, W, C] -> [C, H, W]
img_after = transform(img)

# tensor -> numpy
img_after = img_after.numpy()

# [C, H, W] -> [H, W, C]
img_after = np.transpose(img_after, (1, 2, 0))
plt.imshow(img_after)
plt.axis(False)
plt.tight_layout()
plt.savefig('wolf_after_normalization.jpg')
plt.show()

# print img_after
# print("img_after = ", img_after)
