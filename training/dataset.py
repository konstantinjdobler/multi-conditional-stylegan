# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from typing import Any, Optional
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Disable parallelism for sentence transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        cache_dir   = None,
        condition_dropout_p = 0.0, # Probability for condition dropout. 0 disables the feature.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size and optional label choosing
    ):
        self._name = name
        self._cache_dir = cache_dir
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        # TODO: There is some ambiguity between original labels and raw labels
        self._original_labels = None
        self._condition_dropout_p = condition_dropout_p
        self._mmdc_label_shapes = None
        self._mmdc_condition_order = None
        self._rnd_optional_label = np.random.RandomState(random_seed)

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        if self._use_labels:
            self._raw_labels = self._get_raw_labels()

    @property
    def _label_file(self):
        return os.path.join(self._cache_dir, "labels.pt")

    @property
    def _original_label_file(self):
        return os.path.join(self._cache_dir, "original_labels.json")

    def _get_raw_labels(self):
        if self._raw_labels is None:
            if getattr(self, "_cond_path", None):
                with open(self._cond_path, 'rb') as f:
                    json_dict = json.load(f)
                    self._mmdc_label_shapes = json_dict['shapes']
                    self._mmdc_condition_order = json_dict['condition_order']
            if self._use_labels and self._cache_dir and os.path.isfile(self._label_file):
                self._raw_labels = torch.load(self._label_file)
                with open(self._original_label_file, "r") as f:
                    self._original_labels = json.loads(f.read())
                return self._raw_labels
            
            self._raw_labels, self._original_labels = self._load_raw_labels() if self._use_labels else (None, None)
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)

            if self._use_labels and self._cache_dir and not os.path.isfile(self._label_file):
                print("Save Labels")
                torch.save(self._raw_labels, self._label_file)
                with open(self._original_label_file, "w+") as f:
                    f.write(json.dumps(self._original_labels))
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx, enable_dropout=True)
     
    def dropout_mmdc_condition(self, mmdc_condition, p=0.5, k=2, zero_vector_dropout=True):
        assert self._mmdc_label_shapes != None
        assert self._mmdc_condition_order != None
        # Sample at most k conditions to dropout, each with a probability of p
        dropout_candidate_idxs = list(np.random.choice(len(self._mmdc_condition_order), size=k, replace=False))
        dropout_idxs = []
        for idx in dropout_candidate_idxs:
            if np.random.random() < p:
                dropout_idxs.append(idx)

        curr_condition_start_idx = 0
        for i, condition_type in enumerate(self._mmdc_condition_order):
            label_shape = self._mmdc_label_shapes[i]
            if i in dropout_idxs:
                mmdc_condition[curr_condition_start_idx:curr_condition_start_idx+label_shape] = 0
                if condition_type not in ['keywords', 'emotions'] and not zero_vector_dropout:
                    # `Unknown` token is one-hot encoded at index zero of each condition except for distributions, where we use a zero-vector
                    mmdc_condition[curr_condition_start_idx] = 1
            curr_condition_start_idx += label_shape
        return mmdc_condition

    def _mask_mmdc_condition(self, mmdc_condition, idxs):
        # assert self._mmdc_label_shapes != None
        # assert self._mmdc_condition_order != None
        # Sample at most k conditions to dropout, each with a probability of p
        curr_condition_start_idx = 0
        for i, condition_type in enumerate(self._mmdc_condition_order):
            label_shape = self._mmdc_label_shapes[i]
            if i in idxs:
                mmdc_condition[curr_condition_start_idx:
                               curr_condition_start_idx+label_shape] = 0
            curr_condition_start_idx += label_shape
        return mmdc_condition

    def dropout_mmdc_condition_batch(self, mmdc_condition_batch, p=0.5):

        if np.random.random() < p:
            return mmdc_condition_batch
        for i in range(mmdc_condition_batch.shape[0]):
            num_mask_idxs = random.randint(1, len(self._mmdc_condition_order))
            mask_idxs = list(np.random.choice(
                len(self._mmdc_condition_order), size=num_mask_idxs, replace=False))
            mmdc_condition_batch[i] = self._mask_mmdc_condition(
                mmdc_condition_batch[i], idxs=mask_idxs)
        return mmdc_condition_batch

    def add_noise(self, label, sigma=0.02):
        curr_condition_start_idx = 0
        for i, condition_type in enumerate(self._mmdc_condition_order):
            label_shape = self._mmdc_label_shapes[i]
            if condition_type in ['keywords', 'emotions'] or condition_type.startswith('color'):
                r = np.random.randn(label_shape) * sigma
                label[curr_condition_start_idx:curr_condition_start_idx+label_shape] += r
            curr_condition_start_idx += label_shape
        return label


    @staticmethod
    def _convert_optional_label(optional_labels: np.array, rnd: np.random.RandomState):
        def choose_label(l):
            if len(l) == 1:
                return l[0]
            return l[rnd.randint(len(l))]
        
        if optional_labels.dtype != object:
            return optional_labels

        chosen_labels = list(map(choose_label, optional_labels))
        result = np.concatenate(chosen_labels)
        return np.array(result, dtype=np.float32)

    def get_label(self, idx, enable_dropout=False, enable_noise=False):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        elif label.dtype == object:
            # Multiple options for one labels
            label = self._convert_optional_label(label, self._rnd_optional_label)
        
        if self._condition_dropout_p > 0 and enable_dropout:
            num_potential_drops = max(len(self._mmdc_label_shapes) - 1, 1)
            label = self.dropout_mmdc_condition(label, p=self._condition_dropout_p, k=num_potential_drops)

        if enable_noise:
            label = self.add_noise(label)

        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_original_label(self, idx):
        if self._original_labels == None:
            return None
        idx = self._raw_idx[idx]
        return self._original_labels[idx]

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            elif raw_labels.dtype == object:
                self._label_shape = [np.sum(self._mmdc_label_shapes)]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        cond_path       = None, # Path to external dataset.json. None = fall back to dataset.json in path
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._cond_path = cond_path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        if self._cond_path != None and not os.path.isfile(self._cond_path):
            raise IOError('Path to external dataset.json was defined but does not exist')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        # TODO: hack, filter out images that are not in label file
        if self._cond_path:
            with open(self._cond_path, 'rb') as f:
                json_dict = json.load(f)
                labels = json_dict['labels']
                labels = dict(labels)
            filtered_fnames = [
                name for name in self._image_fnames if labels.get(name) is not None]
            print(
                f"Skipping {len(self._image_fnames) - len(filtered_fnames)} images because they had no corresponding labels.")
            self._image_fnames = filtered_fnames


        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        if self._cond_path != None:
            with open(self._cond_path, 'rb') as f:
                json_dict = json.load(f)
                labels = json_dict['labels']
                shapes = json_dict['shapes']
                condition_order = json_dict['condition_order']
                del json_dict
        else:
            fname = 'dataset.json'
            if fname not in self._all_fnames:
                return None
            with self._open_file(fname) as f:
                labels = json.load(f)['labels']

        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        original_labels = [l for l in labels]
        if self._cond_path != None and shapes != None:
            labels = self.transform_multidomain_conditions(labels, shapes)
        else:
            labels = np.array(labels)
            labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels, original_labels

    @staticmethod
    def _transform_label(label: Any, label_shape: Optional[int]=None, model: Optional[SentenceTransformer]=None):
        # Integers are treated as indices that we need to one-hot encode
        if isinstance(label, int):
            assert label_shape != None
            onehot = np.zeros(label_shape, dtype=np.float32)
            onehot[label] = 1
            return onehot
        # Strings are encoded with BERT
        elif isinstance(label, str):
            assert model != None
            return model.encode(label)
        # [] means there is no label
        elif isinstance(label, list) and len(label) == 0:
            assert label_shape != None
            return np.zeros(label_shape, dtype=np.float32)
        return label

    @staticmethod
    def transform_multidomain_conditions(all_raw_labels, label_shapes, verbose=True):
        processed_labels = []
        if verbose:
            print("Transform Multidomain Conditions")
        
        model = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
        for raw_labels in tqdm(all_raw_labels, disable=not verbose):
            # When using mmdc there always needs to be an array as labels
            # [] means there are no labels
            if len(raw_labels) == 0:
                zeros = np.zeros(np.sum(label_shapes), dtype=np.float32)
                processed_labels.append(zeros)
                continue
            
            labels = []
            assert len(raw_labels) == len(label_shapes)
            # For condition in the raw label
            for i, label_part in enumerate(raw_labels):
                label_shape = label_shapes[i]

                # 0, "str", [0.0, ...] will be put in another array to fit multiple option format
                if not isinstance(label_part, list) or len(label_part) == 0 or isinstance(label_part[0], float):
                    label_part = [label_part]

                optional_labels = list(map(lambda x: ImageFolderDataset._transform_label(x, label_shape, model), label_part))
                assert all([len(l) == label_shape for l in optional_labels])
                labels.append(optional_labels)

            labels = np.array(labels, dtype=object)
            # Create one vector if there is only one option for each label part
            if all([len(part) == 1 for part in labels]):
                labels = list(map(lambda x: x[0], labels))
                labels = np.array(np.concatenate(labels), dtype=np.float32)
                assert len(labels) == np.sum(label_shapes)
            processed_labels.append(labels)
        
        del model
        try:
            return np.array(processed_labels, dtype=np.float32)
        except:
            return np.array(processed_labels, dtype=object)
#----------------------------------------------------------------------------
