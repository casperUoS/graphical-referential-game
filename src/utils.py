from itertools import chain, combinations
import torch, torchvision, random, math, os
import numpy as np

from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid", {'axes.grid' : False})

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("##### DEVICE:",device, flush=True)

path  = '/gpfsscratch/rech/imi/uwv18kd/ebng'
if not (os.path.exists(path)):
    path = os.getcwd()
    print("==============LOCAL==============", flush=True)
else:
    print("==============SERVER==============", flush=True)

##### SYSTEMATIC REFERENTS -> IMAGES & PERSPECTIVES ############################
if not os.path.exists(path+'data'):
    mnist_train = torchvision.datasets.MNIST(root=os.path.join(path,'data'), train=True, download=True, transform=torchvision.transforms.ToTensor())
else:
    mnist_train = torchvision.datasets.MNIST(root=os.path.join(path,'data'), train=True, download=False, transform=torchvision.transforms.ToTensor())


##### DISPLAY FUNCTIONS ########################################################
def show_img(img):
    while img.shape[0] == 1: img = img.squeeze()
    if img.shape[0] == 3: img = img.permute(1,2,0)
    plt.imshow(img.detach().cpu())
    plt.show()

def show_imgs(imgs,labels=None):
    fig, axis = plt.subplots(2 if labels else 1,imgs.shape[0],figsize=(3*imgs.shape[0],3))
    for i in range(imgs.shape[0]):
        to_show = imgs[i].detach().cpu()
        if to_show.shape[0] == 1:
            to_show = to_show.squeeze()
        if labels:
            axis[1][i].imshow(to_show)
            axis[1][i].axis("off")
            axis[0][i].text(0.5,0.1,labels[i],horizontalalignment='center',verticalalignment='center')
            axis[0][i].axis("off")
        else:
            axis[i].imshow(to_show)
            axis[i].axis("off")
    plt.show()


### Sample images of numbers given a list of features (integers)
def sample_nbs(features):
    nb_imgs = []
    for i in features:
        imgs    = mnist_train.data[mnist_train.targets==i]
        idx     = random.randint(0,imgs.shape[0]-1)
        nb_imgs.append(imgs[idx])
    return nb_imgs

### Converts a systematic referent into an image perspective
def convert_to_imgs(dataset,batch_size=1,ood=False):
    dataset_imgs = torch.empty(0,batch_size,1,4*28,4*28)
    for i in range(dataset.shape[0]):
        referent  = dataset[i]
        features  = (referent==1).nonzero(as_tuple=True)[0].tolist()
        #nb_imgs   = sample_nbs(features)

        img_referent = torch.zeros(batch_size,1,4*28,4*28)
        for b in range(batch_size):
            nb_imgs = sample_nbs(features if (not ood) else [feature+5 for feature in features])
            positions = np.random.choice(list(range(0,4*4)),len(features),replace=False)
            for j,position in enumerate(positions):
                x = position%4
                y = math.floor(position/4)
                img_referent[b,0,x*28:x*28+28,y*28:y*28+28] = nb_imgs[j]

        dataset_imgs = torch.cat((dataset_imgs, img_referent.unsqueeze(0)))

    if batch_size==1:    return dataset_imgs.squeeze(1).to(device)
    else:                return dataset_imgs.to(device)

##### MISC #####################################################################
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_random_batch(dataset,batch_size=32):
    N    = len(dataset)
    idxs = np.random.choice(range(N),min(N,batch_size),replace=False)
    return dataset[idxs]
