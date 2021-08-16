import os
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset


class TextDataset(Dataset):
    def __init__(self, config, training=True):
        self.img_w = config['dataset']['img_w']
        self.img_h = config['dataset']['img_h']
        self.training = training
        self.case_sensitive = config['dataset']['case_sensitive']
        self.to_gray = config['dataset']['to_gray']
        self.transform = config['dataset']['transform']
        self.target_transform = config['dataset']['target_transform']
        self.all_images = []
        self.all_labels = []

        if training:
            images, labels = self.get_base_info(config['dataset']['txt_file'], config['dataset']['img_root'])
            self.all_images += images
            self.all_labels += labels
        else:
            imgs = os.listdir(config['dataset']['img_root'])
            for img in imgs:
                self.all_images.append(os.path.join(config['dataset']['img_root'], img))

    def get_base_info(self, txt_file, img_root):
        image_names = []
        labels = []
        with open(txt_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                image_name = line[0]
                label = '\n'.join(line[1:])
                if len(label) > LabelTransformer.max_length and LabelTransformer.max_length != -1:
                    continue
                image_name = os.path.join(img_root, image_name)
                image_names.append(image_name)
                labels.append(label)
        return image_names, labels

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        file_name = self.all_images[idx]
        img = Image.open(file_name)
        try:
            if self.to_gray:
                img = img.convert('L')
            else:
                img = img.convert('RGB')
        except Exception as e:
            print('Error image for {}'.format(file_name))

        if self.transform is not None:
            img, width_ratio = self.transform(img)

        if self.training:
            label = self.all_labels[idx]
            if self.target_transform is not None:
                label = self.target_transform(label)

            if not self.case_sensitive:
                label = label.lower()
            return img, label
        else:
            return img, file_name
