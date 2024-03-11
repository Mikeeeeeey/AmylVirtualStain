import os
import time
import random
import numpy as np
import torch.utils.data as data
import torch
import threading
import queue
from dataclasses import dataclass, field
from typing import Any
import torchvision.transforms as transforms


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


def rgb_to_ycbcr(label_temp):
    # range input_rgb:0-255, output_ycbcr: 0-255
    label_temp = label_temp / 255
    label = label_temp.copy()
    label[0, :, :] = (
        16
        + 65.481 * label_temp[0, :, :]
        + 128.533 * label_temp[1, :, :]
        + 24.966 * label_temp[2, :, :]
    )
    label[1, :, :] = (
        128
        - 37.797 * label_temp[0, :, :]
        - 74.203 * label_temp[1, :, :]
        + 112 * label_temp[2, :, :]
    )
    label[2, :, :] = (
        128
        + 112 * label_temp[0, :, :]
        - 93.786 * label_temp[1, :, :]
        - 18.214 * label_temp[2, :, :]
    )

    return label


def normalize_input(im):
    # norm_factor = [1 / 3000, 1 / 1500, 1 / 2000, 1 / 300]
    norm_factor = [1/3000, 1/3000, 1/3000, 1/2000]
    for ch_idx in range(len(norm_factor)):
        im[ch_idx, :, :] = im[ch_idx, :, :] * norm_factor[ch_idx] *2
    return im


# def normalize_input(im):
#     store_mean = [1407.40556465, 641.48790788, 874.29439001, 227.81690292]
#     store_std = [1327.55166207, 578.59555949, 840.62231557, 95.48584565]
#     for ch_idx in range(im.shape[0]):
#         im[ch_idx, :, :] = (im[ch_idx, :, :] - store_mean[ch_idx]) / store_std[ch_idx]
#     return im


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate"""

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


class patch_generator:
    def __init__(
        self,
        image_dir,
        image_size,
        is_training,
        color_space,
        patch_per_tile,
        raw_downsample,
    ):
        self.image_dir = image_dir
        self.image_filenames = self.get_label_list(image_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.color_space = color_space
        self.patch_per_tile = patch_per_tile
        self.raw_downsample = raw_downsample

        self.init_count = 0
        self.lock = threading.Lock()
        self.yield_lock = threading.Lock()
        self.path_id_generator = threadsafe_iter(
            self.get_path_i(len(self.image_filenames))
        )
        self.inputs = []
        self.labels = []

    def __iter__(self):
        while True:
            # In the start of each epoch we shuffle the data paths
            with self.lock:
                if self.init_count == 0:
                    # random.seed(time.time())
                    random.shuffle(self.image_filenames)
                    self.inputs, self.labels, self.batch_paths = [], [], []
                    self.init_count = 1
            # Iterates through the input paths in a thread-safe manner
            for path_id in self.path_id_generator:
                input_img, label_img = self.get_img_at(path_id)
                if len(input_img) > 0:
                    with self.yield_lock:
                        yield input_img, label_img
            # At the end of an epoch we re-init data-structures
            with self.lock:
                self.init_count = 0

    def get_path_i(self, paths_count):
        """Cyclic generator of paths indice"""
        current_path_id = 0
        while True:
            yield current_path_id
            current_path_id = (current_path_id + 1) % paths_count

    def get_label_list(self, image_dir):
        return [
            os.path.join(image_dir, x)
            for x in os.listdir(image_dir)
            if self.is_target_file(x)
        ]

    def is_target_file(self, filename):
        return filename.endswith(".npy")

    def is_good_fov(self, im):
        return np.mean(im) < 255

    def load_img(self, filepath):
        y = np.load(filepath).astype(np.float32)
        # y = y[:,:,1]
        return y

    def get_img_at(self, index):
        path = self.image_filenames[index]
        target_img_pol = self.load_img(path).astype("float32")
        target_img_bf = self.load_img(path.replace("target-Pol", "target-BF")).astype("float32")
        target_img_full = np.concatenate((target_img_bf,target_img_pol), axis=0)
        input_img_full = self.load_img(path.replace("target-Pol", "input-AF")).astype(
            "float32"
        )
        input_img_full = normalize_input(input_img_full)
        assert input_img_full.shape[-2:-1] == target_img_full.shape[-2:-1]
        assert len(input_img_full.shape) == 3
        input_img_full = input_img_full[:, :, :]
        if input_img_full.ndim == 2:
            input_img_full = np.expand_dims(input_img_full, axis=0)
        if target_img_full.ndim == 2:
            target_img_full = np.expand_dims(target_img_full, axis=0)

        cropEdge = 10
        crop_image_size = self.image_size * self.raw_downsample

        # augumentation
        if self.is_training:
            s = np.floor(crop_image_size * 1.2).astype("int16")
            r = random.randint(0, 3)
            input_img_full = np.rot90(input_img_full, r, axes=(1, 2))
            target_img_full = np.rot90(target_img_full, r, axes=(1, 2))
            if random.randint(0, 1):
                input_img_full = input_img_full[:, :, ::-1]
                target_img_full = target_img_full[:, :, ::-1]
            if random.randint(0, 1):
                input_img_full = input_img_full[:, ::-1, :]
                target_img_full = target_img_full[:, ::-1, :]

            for ii in range(input_img_full.shape[0]):
                input_img_full[ii, :, :] = input_img_full[ii, :, :] * random.uniform(
                    0.95, 1.05
                )
        else:
            s = crop_image_size

        images, labels = [], []
        for _ in range(self.patch_per_tile):
            got_image = False
            xx = random.randint(cropEdge, input_img_full.shape[1] - s - cropEdge)
            yy = random.randint(cropEdge, input_img_full.shape[2] - s - cropEdge)
            if self.is_good_fov(target_img_full[:, xx : xx + s, yy : yy + s]):
                got_image = True

            if got_image:
                input_img = input_img_full[:, xx : xx + s, yy : yy + s].copy()
                target_img = target_img_full[:, xx : xx + s, yy : yy + s].copy()
                if self.color_space == "YCbCr":
                    target_img = rgb_to_ycbcr(target_img)
                    target_img = target_img / 255 * 2 - 1
                elif self.color_space == "RGB":
                    temp1 = target_img[0:3,:,:] / 128 - 1
                    temp2 = target_img[3:7,:,:] * 2 - 1
                    target_img = np.concatenate((temp1,temp2),axis=0)
                input_img = torch.tensor(input_img)
                target_img = torch.tensor(target_img)

                if self.is_training:
                    (
                        angle,
                        translations,
                        scale,
                        shear,
                    ) = transforms.RandomAffine.get_params(
                        degrees=[0, 0],
                        translate=None,
                        scale_ranges=(0.95, 1.05),
                        shears=(-10, 10, -10, 10),
                        img_size=(s, s),
                    )
                    input_img = transforms.functional.affine(
                        input_img,
                        angle=angle,
                        translate=translations,
                        scale=scale,
                        shear=shear,
                    )
                    target_img = transforms.functional.affine(
                        target_img,
                        angle=angle,
                        translate=translations,
                        scale=scale,
                        shear=shear,
                    )

                    input_img = transforms.functional.center_crop(
                        input_img, [crop_image_size, crop_image_size]
                    )
                    target_img = transforms.functional.center_crop(
                        target_img, [crop_image_size, crop_image_size]
                    )

                if self.raw_downsample > 1:
                    input_img = torch.nn.functional.avg_pool2d(
                        input_img, self.raw_downsample
                    )
                    target_img = torch.nn.functional.avg_pool2d(
                        target_img, self.raw_downsample
                    )
                # print(target_img.shape)

                images.append(input_img)
                labels.append(target_img)

        return images, labels


class Dataset_train(data.Dataset):
    def __init__(
        self,
        image_dir,
        image_size,
        n_workers=1,
        epoch_len=1000,
        is_training=True,
        color_space="RGB",
        queue_size=1000,
        patch_per_tile=10,
        raw_downsample=1,
    ):
        super(Dataset_train, self).__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_training = is_training
        self.color_space = color_space
        self.patch_per_tile = patch_per_tile
        self.raw_downsample = raw_downsample

        self.queue_size = queue_size
        self.queue = queue.PriorityQueue(maxsize=self.queue_size)
        self.train_thread_killer = thread_killer()
        self.train_thread_killer.set_tokill(False)
        self.n_workers = n_workers
        self.threads = []
        self.epoch_len = epoch_len
        self.start_threads(self.n_workers)

    def __getitem__(self, idx):
        return self.queue.get(block=True).item

    def __len__(self):
        return self.epoch_len

    def start_threads(self, n_workers):
        for n in range(n_workers):
            t = threading.Thread(
                target=self.batch_feeder,
                args=(
                    n,
                    self.train_thread_killer,
                    self.queue,
                    patch_generator(
                        self.image_dir,
                        self.image_size,
                        self.is_training,
                        self.color_space,
                        self.patch_per_tile,
                        self.raw_downsample,
                    ),
                ),
            )
            self.threads.append(t)
            t.start()

    def batch_feeder(self, idx, tokill, batches_queue, dataset_generator):
        """
        Threaded worker for pre-processing input data.
        tokill is a thread_killer object that indicates whether a thread
        should be terminated
        dataset_generator is the training/validation dataset generator
        batches_queue is a limited size thread-safe Queue instance.
        """
        random.seed(idx)
        while tokill() is False:
            for _, (batch_images, batch_labels) in enumerate(dataset_generator):
                # fill the queue with new batch until reaching the max size.
                for i in range(len(batch_images)):
                    batches_queue.put(
                        PrioritizedItem(
                            random.randint(1, self.queue_size),
                            (batch_images[i], batch_labels[i]),
                        ),
                        block=True,
                    )
                    time.sleep(1e-3)
                if tokill() is True:
                    return


class Dataset_test(data.Dataset):
    def __init__(self, image_dir, raw_downsample=1):
        super(Dataset_test, self).__init__()
        self.image_dir = image_dir
        self.image_filenames = self.get_tile_list(image_dir)
        self.raw_downsample = raw_downsample

    def __getitem__(self, index):
        tile_id = self.get_tile_name(self.image_filenames[index])
        input_img = np.load(os.path.join(self.image_dir.replace('output-model_12_6','input'), tile_id + ".npy")).astype(
            "float32"
        )
        if input_img.ndim == 2:
            input_img = np.expand_dims(input_img, axis=0)
        assert len(input_img.shape) == 3
        input_img = normalize_input(input_img)
        input_img = input_img[:, :, :]
        input_img = torch.tensor(input_img)
        if self.raw_downsample > 1:
            input_img = torch.nn.functional.avg_pool2d(input_img, self.raw_downsample)
        return input_img, tile_id

    def get_tile_list(self, image_dir):
        return [x for x in os.listdir(image_dir) if self.is_target_file(x)]

    def __len__(self):
        return len(self.image_filenames)

    def get_tile_name(self, image_file_name):
        return image_file_name.split(".")[0]

    def is_target_file(self, filename):
        return filename.endswith(".png")
