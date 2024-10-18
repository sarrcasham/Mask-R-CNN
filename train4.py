import os
import json
import numpy as np
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class DOGDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Add class
        self.add_class("dataset", 1, "dog")

        # Define data locations
        images_dir = os.path.join(dataset_dir, 'images')
        annots_file = os.path.join(dataset_dir, 'annots.json')

        # Load annotations
        with open(annots_file, 'r') as f:
            annots = json.load(f)
            
        print(annots.keys())
        for annot in annots['annotations']:
            image_id = annot['image_id']
            file_name = [img['file_name'] for img in annots['images'] if img['id'] == image_id][0]

            if is_train and annots['images'].index(next(img for img in annots['images'] if img['id'] == image_id)) >= 150:
                continue
            if not is_train and annots['images'].index(next(img for img in annots['images'] if img['id'] == image_id)) < 150:
                continue

            img_path = os.path.join(images_dir, file_name)
            ann_path = os.path.join(dataset_dir, 'annots', f'{image_id}.json')
            self.add_image('dataset', image_id=image_id, path=img_path, annot=ann_path)

    def extract_boxes(self, filename):
        with open(filename) as f:
            data = json.load(f)

        boxes = []
        for annot in data['annots']:
            xmin = min([point[0] for point in annot['segmentation']])
            ymin = min([point[1] for point in annot['segmentation']])
            xmax = max([point[0] for point in annot['segmentation']])
            ymax = max([point[1] for point in annot['segmentation']])
            boxes.append([xmin, ymin, xmax, ymax])

        width = data['images'][0]['width']
        height = data['images'][0]['height']
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annot']
        boxes, w, h = self.extract_boxes(path)

        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('dog'))

        return masks, np.asarray(class_ids, dtype='int32')

# Define the configuration
class DOGConfig(Config):
    NAME = "dog_cfg"
    NUM_CLASSES = 1 + 1  # Background + dog class
    STEPS_PER_EPOCH = 100  # Adjust based on your dataset size

# Prepare the train and validation sets
dataset_dir = r"C:\Users\Asarv\Documents\office\dataset"

train_set = DOGDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()

val_set = DOGDataset()
val_set.load_dataset(dataset_dir, is_train=False)
val_set.prepare()

# Load and train the model
config = DOGConfig()
model = MaskRCNN(mode='training', model_dir='./', config=config)

# Load weights (COCO pre-trained weights or your own)
model.load_weights('path/to/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model
model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')