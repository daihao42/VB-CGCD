
from utils import dataloader
from utils.dataloader import ClassIncrementalLoader

class ImageNet100Loader():
    def __init__(self, args):
        self.args = args

    def makeT5Loader(self):
        base = 50
        increment = 10
        num_labeled = 50000
        num_novel_inc = 1000
        num_known_inc = 60

        loader = dataloader.StrictPerClassIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_inc=num_novel_inc, num_known_inc=num_known_inc)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader

    def makeT10Loader(self):
        base = 50
        increment = 5
        num_labeled = 50000
        num_novel_inc = 1000
        num_known_inc = 60

        loader = dataloader.StrictPerClassIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_inc=num_novel_inc, num_known_inc=num_known_inc)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader


    # Vin-comparison
    def makeVinLoader(self):
        base = 10
        increment = 10
        num_labeled = 10000
        num_novel_inc = 1000
        num_known_inc = 60

        loader = dataloader.StrictPerClassIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_inc=num_novel_inc, num_known_inc=num_known_inc)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader

