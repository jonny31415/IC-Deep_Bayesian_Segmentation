import tensorflow as tf
import numpy as np
import functools
import random
from datasets import DatasetBase, dataset_mean, dataset_std
from tensorflow.keras.preprocessing.image import load_img
from utils import print_np, PIL2numpy


class CityscapesGenerator(tf.keras.utils.Sequence, DatasetBase):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, data_aug = False, add_ignore_class=False, crop_size=(512,512)):
        super().__init__()

        self.n_classes = 19

        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.data_aug = data_aug
        self.add_ignore_class = add_ignore_class
        self.crop_size = crop_size
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1] # not to train
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        self.y_20 = None

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)  # Returns PIL image
            img_np = PIL2numpy(img)

            # Data augmentation
            if self.data_aug:
                img_np = self.color_jitter(img_np, 0.2, 0.2, 0.2, 0.1) 
            
            # Normalize image
            img_norm = self.normalize(img_np)
            x[j] = img_norm

        y_temp = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y = np.zeros((self.batch_size,) + self.img_size + (self.n_classes+1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            label = load_img(path, target_size=self.img_size, color_mode="grayscale")  # Returns PIL image
            label_np = PIL2numpy(label)

            y_temp[j] = np.expand_dims(label_np, 2)  # (h, w, 1)
            y_temp[j] = self.fix_indxs(y_temp[j])
            y[j] = self.one_hot_encode(y_temp[j])

        self.y_20 = y.copy()
        if not self.add_ignore_class:
            y = y[:, :, :, :19]

        if self.data_aug:
            x, y = self.random_crop(x, y)

        x = tf.convert_to_tensor(x)
        y = tf.cast(tf.convert_to_tensor(y), tf.float32)

        return x, y

    def fix_indxs(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        mask[mask == 255] = len(self.valid_classes)
        return mask

    def one_hot_encode(self, lbl):
        new_lbl = np.array(self.get_one_hot(lbl.reshape(-1),self.n_classes+1))
        new_lbl = new_lbl.reshape(self.img_size[0], self.img_size[1], self.n_classes+1)
        return new_lbl

    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])

    @staticmethod
    def from_one_hot_to_rgb_bkup(image):

        palette = [[128, 64, 128],
                   [244, 35, 232],
                   [70, 70, 70],
                   [102, 102, 156],
                   [190, 153, 153],
                   [153, 153, 153],
                   [250, 170, 30],
                   [220, 220, 0],
                   [107, 142, 35],
                   [152, 251, 152],
                   [70, 130, 180],
                   [220, 20, 60],
                   [255, 0, 0],
                   [0, 0, 142],
                   [0, 0, 70],
                   [0, 60, 100],
                   [0, 80, 100],
                   [0, 0, 230],
                   [119, 11, 32],
                   [0, 0, 0]]

        _, W, H, _ = image.shape
        palette = tf.constant(palette, dtype=tf.uint8)
        class_indexes = tf.argmax(image, axis=-1)

        class_indexes = tf.reshape(class_indexes, [-1])
        color_image = tf.gather(palette, class_indexes)
        color_image = tf.reshape(color_image, [-1, W, H, 3])

        color_image = color_image.numpy().astype(np.uint8)

        return color_image
    
    def normalize(self, img_np):
        # Normalize image
        input_data = np.array(img_np)/255
        layer = tf.keras.layers.experimental.preprocessing.Normalization(mean=dataset_mean, variance=np.square(dataset_std))
        img = layer(input_data).numpy()
        return img

    def color_jitter(self, image, brightness=0, contrast=0, saturation=0, hue=0):
        """Color jitter.

        Examples
        --------
        >>> color_jitter(img, 25, 0.2, 0.2, 0.1)

        """
        tforms = []
        if brightness > 0:
            tforms.append(functools.partial(tf.image.random_brightness, max_delta=brightness))
        if contrast > 0:
            tforms.append(functools.partial(tf.image.random_contrast, lower=max(0, 1 - contrast), upper=1 + contrast))
        if saturation > 0:
            tforms.append(functools.partial(tf.image.random_saturation, lower=max(0, 1 - saturation), upper=1 + saturation))
        if hue > 0:
            tforms.append(functools.partial(tf.image.random_hue, max_delta=hue))

        random.shuffle(tforms)
        for tform in tforms:
            image = tform(image)

        return image 

    def random_crop(self, img, label):
        # img: (batch, height, width, channels)
        assert img.shape[3] == 3
        img_croped = np.zeros((img.shape[0],)+self.crop_size+(img.shape[3],))
        label_croped = np.zeros((label.shape[0],)+self.crop_size+(label.shape[3],))
        height, width = img.shape[1], img.shape[2]
        dx, dy = self.crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        for i in range(img.shape[0]):
            img_croped[i] = img[i][y:(y+dy), x:(x+dx), :]
            label_croped[i] = label[i][y:(y+dy), x:(x+dx), :]
        return img_croped, label_croped
