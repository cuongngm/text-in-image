import os
import cv2


class VietnameseDataset:
    def __init__(self, img_dir, label_dir):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir

    def rename(self):
        for filename in os.listdir(self.img_dir):
            idx_name = filename[2:6]
            idx_name = int(idx_name)
            os.rename(os.path.join(self.img_dir, filename), os.path.join(self.img_dir, str(idx_name) + '.jpg'))

    def visualize(self):
        for filename in os.listdir(self.img_dir):
            img_path = os.path.join(self.img_dir, filename)
            gt_path = 'gt_' + filename.replace('.jpg', '.txt')
            img = cv2.imread(img_path)
            with open(os.path.join(self.label_dir, gt_path), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    poly = line.strip().split(',')
                    poly = poly[:8]
                    poly = list(map(int, poly))
                    cv2.rectangle(img, (poly[0], poly[1]), (poly[4], poly[5]), (0, 255, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            break


if __name__ == '__main__':
    dataset = VietnameseDataset(img_dir='../../dataset/vietnamese/unseen_test_images',
                                label_dir='../../dataset/vietnamese/labels')
    dataset.visualize()