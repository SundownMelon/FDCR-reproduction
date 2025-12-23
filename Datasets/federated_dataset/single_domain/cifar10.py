import torch
from PIL import Image
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from Datasets.federated_dataset.single_domain.utils.single_domain_dataset import SingleDomainDataset
from utils.conf import single_domain_data_path


class MyCIFAR10(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False, data_name=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.dataset = self.__build_truncated_dataset__()
        self.data = self.dataset.data

        if hasattr(self.dataset, 'labels'):
            self.targets = self.dataset.labels
        elif hasattr(self.dataset, 'targets'):
            self.targets = self.dataset.targets

        if isinstance(self.targets, torch.Tensor):
            self.targets = self.targets.numpy()
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.numpy()

    def __build_truncated_dataset__(self):
        dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        return dataobj

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class FLCIFAR10(SingleDomainDataset):
    NAME = 'fl_cifar10'
    SETTING = 'label_skew'
    N_CLASS = 10

    def __init__(self, args, cfg) -> None:
        super().__init__(args, cfg)

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    def get_data_loaders(self):
        pri_aug = self.cfg.DATASET.aug
        if pri_aug == 'weak':
            train_transform = self.train_transform
        elif pri_aug == 'strong':
            train_transform = self.train_transform

        train_dataset = MyCIFAR10(root=single_domain_data_path(), train=True,
                                  download=True, transform=train_transform)

        test_dataset = MyCIFAR10(single_domain_data_path(), train=False,
                                 download=True, transform=self.test_transform)
        
        self.partition_label_skew_loaders(train_dataset, test_dataset)
