from torchvision.transforms import functional as F
import torch, sys
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join

def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

# load 10 real images
mypath = sys.argv[1]
# mypath="./cityscapes"
onlyfiles = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]
real_images = []
for n in range(0, len(onlyfiles)):
  real_images.append(Image.fromarray(np.array(Image.open( join(mypath,onlyfiles[n]) ))))
real_images = torch.cat([preprocess_image(image) for image in real_images])

# load 10 fake image randomly
mypath=sys.argv[2]
# mypath="./dreambooth_output"
onlyfiles = [ f for f in sorted(listdir(mypath)) if isfile(join(mypath,f)) ]
fake_images = []
for n in range(0, len(onlyfiles)):
  fake_images.append(Image.fromarray(np.array(Image.open( join(mypath,onlyfiles[n]) ))))
fake_images = torch.cat([preprocess_image(image) for image in fake_images])

# FID
from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")

