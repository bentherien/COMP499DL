from numpy.random import RandomState
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset

from nag_trainer import NAGTrainer
import collections
import torchvision

OptParams = collections.namedtuple('OptParams', 'lr factor ' +
                                   'batch_size epochs ' +
                                   'decay_epochs decay_rate gamma')
OptParams.__new__.__defaults__ = (None, None, None, None, None, None, None)

NAGParams = collections.namedtuple('NAGParams',
                                   'nz force_l2 is_pixel z_init is_classifier disc_net loss data_name noise_proj shot')
NAGParams.__new__.__defaults__ = (None, None, None, None)

from interpolate import interpolate_points
def getGenImage(z1,z2,gen):
    steps = steps
    interp = interpolate_points(z1,z2,n_steps=steps, slerp=True, print_mode=True)
    candidate = interp[int(np.rand() *steps),:]
    im = gen(candidate)
    return im




def runGlico(train_labeled_dataset, classes,epochs):
    FILE_NAME = "tester.py_cifar100_test_run"

    nag_params = NAGParams(nz=512, force_l2=False, is_pixel=True, z_init='rndm', 
    is_classifier=True, disc_net='conv', loss='ce', data_name='cifar-10', noise_proj=True, shot=0)

    nag_opt_params = OptParams(lr=0.001, factor=0.7, batch_size=128, epochs=epochs, 
    decay_epochs=70, decay_rate=0.5, gamma=0.5)


    nt = NAGTrainer(dataset=[train_labeled_dataset, [], []], nag_params=nag_params, rn=FILE_NAME, resume=False,
                                num_classes=classes)

    nt.train_test_nag(nag_opt_params)

    return nt

if __name__ == "__main__":
    data_dir = '../../data'
    cifar_dir_cs = '~/cs/dataset/CIFAR/'
    cifar_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    seed = 1
    prng = RandomState(seed)
    random_permute = prng.permutation(np.arange(0, 5000))

    indx_train = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute[0:10]] for classe in range(0, 10)])
    train_labeled_dataset = Subset(cifar_data, indx_train) #new dataset



    classes = len(set(cifar_data.targets))
    print(len(cifar_data.data),cifar_data.data.shape)
    runGlico(train_labeled_dataset, classes)