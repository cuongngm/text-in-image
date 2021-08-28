import yaml
import matplotlib.pyplot as plt
import cv2
import numpy as np
from src.loader.detection.det_loader import DetLoader
from src.utils.utils_function import read_json

with open('config/db_resnet50.yaml', 'r') as stream:
    cfg = yaml.safe_load(stream)
"""
train_dataset = DetLoader(cfg, is_training=False)
# train_loader = DataLoader(train_dataset, shuffle=True, batch_size=4)
# samples = next(iter(train_dataset))
samples = train_dataset[23]
img = samples['img']

gt = samples['gt']
gt_mask = samples['gt_mask']
thresh_map = samples['thresh_map']
thresh_mask = samples['thresh_mask']
print(img.shape)
fig = plt.figure(figsize=(10, 10))
rows = 2
columns = 2
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("First")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
# showing image
plt.imshow(gt)
plt.axis('off')
plt.title("Second")

# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
# showing image
plt.imshow(thresh_map)
plt.axis('off')
plt.title("Third")

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)
# showing image
plt.imshow(thresh_mask)
plt.axis('off')
plt.title("Fourth")
# plt.imshow(img)

plt.show()

polys = samples['polys']
for poly in polys:
    poly = np.array(poly)
    poly = poly.reshape((-1, 1, 2))
    img = cv2.polylines(img, np.int32([poly]), isClosed=True, color=(255, 0, 0), thickness=2)
plt.imshow(img)
plt.show()
"""
