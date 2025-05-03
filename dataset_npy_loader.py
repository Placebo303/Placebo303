# ✅ 全新改写版 dataset_npy_loader.py （支持LFP-HR为主控，Stack_normalized容错匹配）

import os
import numpy as np

class Dataset:
    def __init__(self, Target3D_path, Synth_view_path, LFP_path, n_num, lf2d_base_size,
                 n_slices=41, shuffle_for_epoch=True, **kwargs):
        self.n_slices = n_slices
        self.Target3D_path = Target3D_path
        self.Synth_view_path = Synth_view_path
        self.LFP_path = LFP_path
        self.n_num = n_num
        self.lf2d_base_size = lf2d_base_size
        self.shuffle_for_epoch = shuffle_for_epoch
        self.sample_ratio = 1.0

        self.update_parameters(allow_new=True, **kwargs)

    def update_parameters(self, allow_new=False, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if attr_new:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def _get_npy_list(self, path):
        npy_list = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
        list_len = int(len(npy_list) * self.sample_ratio)
        return npy_list[:list_len]

    def prepare(self, batch_size, n_epochs):
        if not (os.path.exists(self.LFP_path) and os.path.exists(self.Synth_view_path)):
            raise Exception('[❌] Data path not found!')

        self.SynthView_list = self._get_npy_list(self.Synth_view_path)
        self.LFP_list = self._get_npy_list(self.LFP_path)
        self.Target3D_list = self._get_npy_list(self.Target3D_path)

        if len(self.SynthView_list) == 0 or len(self.LFP_list) == 0:
            raise Exception("[❌] HR or LR loading failed, check file paths!")

        assert len(self.SynthView_list) == len(self.LFP_list)

        self.total_data_num = len(self.SynthView_list)
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.test_img_num = max(1, int(self.total_data_num * 0.1))
        self.train_img_num = self.total_data_num - self.test_img_num

        self.test_SynthView_list = self.SynthView_list[:self.test_img_num]
        self.test_LFP_list = self.LFP_list[:self.test_img_num]

        self.train_SynthView_list = self.SynthView_list[self.test_img_num:]
        self.train_LFP_list = self.LFP_list[self.test_img_num:]

        self.training_pair_num = self.train_img_num

        print(f"✅ Train Samples: {self.training_pair_num}, Test Samples: {self.test_img_num}")

        self.data_shuffle_matrix = []
        for idx in range(self.n_epochs+1):
            temp = np.arange(self.training_pair_num, dtype=np.int32)
            if self.shuffle_for_epoch:
                temp = np.random.permutation(temp)
            else:
                temp.sort()
            self.data_shuffle_matrix.append(temp)
        self.data_shuffle_matrix = np.stack(self.data_shuffle_matrix, axis=0)

        self.cursor = 0
        self.epoch = 0

        return self.training_pair_num

    def _load_npy_batch(self, SynthView_paths, LFP_paths):
        Stack_batch = []
        HR_batch = []
        LF_batch = []

        for h_path, l_path in zip(SynthView_paths, LFP_paths):
            hr = np.load(os.path.join(self.Synth_view_path, h_path))
            print(f"[DEBUG] Loaded HR npy {h_path}, shape = {hr.shape}")
            lf = np.load(os.path.join(self.LFP_path, l_path))
            print(f"[DEBUG] Loaded LF npy {l_path}, shape = {lf.shape}")
            HR_batch.append(hr)
            LF_batch.append(lf)

            # 尝试找对应的Target3D
            base_name = os.path.splitext(h_path)[0]
            stack_file = base_name + '.npy'
            stack_path = os.path.join(self.Target3D_path, stack_file)

            if os.path.exists(stack_path):
                stack = np.load(stack_path)
            else:
                print(f"[⚠️] Warning: Target3D not found for {stack_file}, using zeros.")
                expected_shape = (hr.shape[0]*3, hr.shape[1]*3, self.n_slices)
                stack = np.zeros(expected_shape, dtype=hr.dtype)

            Stack_batch.append(stack)

        return np.array(Stack_batch), np.array(HR_batch), np.array(LF_batch)

    def for_test(self):
        return self._load_npy_batch(self.test_SynthView_list, self.test_LFP_list)

    def hasNext(self):
        return self.epoch < self.n_epochs

    def iter(self):
        if self.epoch >= self.n_epochs:
            raise Exception('epoch index out of bounds:%d/%d' %(self.epoch, self.n_epochs))

        if self.cursor + self.batch_size > self.training_pair_num:
            self.epoch += 1
            self.cursor = 0
            if self.epoch >= self.n_epochs:
                raise Exception('epoch index out of bounds:%d/%d' %(self.epoch, self.n_epochs))

        idx = self.cursor
        end = idx + self.batch_size
        shuffle_idx = self.data_shuffle_matrix[self.epoch][idx:end]

        self.cursor += self.batch_size

        SynthView_paths = [self.train_SynthView_list[i] for i in shuffle_idx]
        LFP_paths = [self.train_LFP_list[i] for i in shuffle_idx]

        return self._load_npy_batch(SynthView_paths, LFP_paths) + (idx, self.epoch)
