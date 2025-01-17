import os
import logging

import h5py
import torch

from timing import exec_time



class MemoryIterator:

    def __init__(self, fpath, names):
        if isinstance(names, str):
            names = [names]

        with h5py.File(fpath, "r") as file:
            self.groups = file.attrs["groups"]
            self.group_size = file.attrs["group_size"]
            self.names = names
            self.last_group_idx = file[f"{self.groups}"].attrs["data_pointer"]

            if self.last_group_idx > self.group_size:
                raise ValueError(f"Invalid group size, expected {self.group_size} got {self.last_group_idx}")

            for name in names:
                if name not in file.attrs["names"]:
                    raise ValueError(f"Could not find key {name}, expected one of {self.names}")
                
        self.curr_group = 1
        self.curr_idx = 0
        self.path = fpath

    def __enter__(self):
        self.file = h5py.File(self.path, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.curr_group == self.groups and self.curr_idx > self.last_group_idx:
            raise StopIteration

        items = []
        for name in self.names:
            items.append(self.file[f"{self.curr_group}/{name}"][self.curr_idx])

        self.curr_idx += 1
        if self.curr_idx == self.group_size:
            self.curr_group += 1
            self.curr_idx = 0

        return items if len(items) > 1 else items[0]


class DiskMemory:

    def __init__(self, file, shape_dict, dtype_dict, data_transform, max_size=2**16):
        try:
            with h5py.File(file, "r") as fout:
                pass
        except FileNotFoundError:
            os.makedirs(os.path.abspath(os.path.join(file, "..")), exist_ok=True)

            with h5py.File(file, "w-") as fout:
                fout.attrs["groups"] = 0
                fout.attrs["group_size"] = max_size
                fout.attrs["size"] = 0
                fout.attrs["names"] = list(shape_dict.keys())
                for key in shape_dict:
                    fout.attrs[f"{key}_shape"] = shape_dict[key]
                    fout.attrs[f"{key}_dtype"] = dtype_dict[key]

                DiskMemory._add_group(fout)

        self.file = file
        self.transform = data_transform

    @exec_time
    def add(self, entries):
        N = len(entries)

        with h5py.File(self.file, "r+") as fout:
            max_size = fout.attrs["group_size"]

            group_id = fout.attrs["groups"]
            group = fout[f"{group_id}"]

            data_pointer = group.attrs["data_pointer"]
            idx = data_pointer

            for i in range(N):
                exp = self.transform(entries[i])
                for key in fout.attrs["names"]:
                    group[key][idx+1] = exp[key]

                idx += 1

                if idx == max_size-1:
                    group.attrs["data_pointer"] = max_size - 1
                    fout.attrs["size"] += idx - data_pointer
                    group = DiskMemory._add_group(fout)

                    data_pointer = group.attrs["data_pointer"]
                    idx = data_pointer

            group.attrs["data_pointer"] = idx
            fout.attrs["size"] += idx - data_pointer

    @staticmethod
    def _add_group(file):
        new_group = file.attrs["groups"] + 1
        max_size = file.attrs["group_size"]

        group = file.create_group(f"{new_group}")
        group.attrs["data_pointer"] = -1
        for name in file.attrs["names"]:
            shape = file.attrs[f"{name}_shape"]
            dtype = file.attrs[f"{name}_dtype"]
            group.create_dataset(name, shape=(max_size, *shape), dtype=dtype, compression="gzip", compression_opts=6)

        file.attrs["groups"] = new_group

        return group


class Memory:

    def __init__(self, size, alpha, beta, min_prob, beta_decay):
        self.size = size
        self.data = [None for _ in range(size)]
        self.priorities = torch.zeros(self.size)
        self.new_data = []
        self.data_index = 0
        self.full = False
        self.alpha = alpha
        self.beta = beta
        self.beta_decay = beta_decay
        self.min_prob = min_prob

    def sample(self, num_samples, shuffle=True):
        probs = self.priorities / torch.sum(self.priorities)

        if len(self.new_data) > 0:
            idxs = self.new_data[:num_samples]
            del self.new_data[:num_samples]
            
            weights = torch.ones(len(idxs))

            if len(idxs) < num_samples:
                logging.warning(f"Not enough new data for full batch, expected {num_samples} new entries, got {len(idxs)}")
        else:
            idxs = torch.multinomial(probs, num_samples=num_samples, replacement=False).tolist()
            weights = (1/(probs[idxs]*self.size))**self.beta
            self.beta *= self.beta_decay
            self.beta = min(1.0, self.beta)

        if shuffle:
            shuffled_idx = torch.randperm(num_samples)
            weights = weights[shuffled_idx]
            idxs = torch.tensor(idxs)[shuffled_idx].tolist()

        batch = []
        for i in idxs:
            batch.append(self.data[i])

        return batch, idxs, weights

    def _safe_add(self, entries):
        new_index = len(entries) + self.data_index

        self.new_data = [i for i in range(self.data_index, new_index)] + self.new_data

        self.data[self.data_index:new_index] = entries
        self.data_index = new_index if new_index < self.size else 0

    def add(self, entries):
        if len(entries) > self.size:
            entries = entries[-self.size:]

        new_index = len(entries) + self.data_index

        if not self.full and new_index >= self.size:
            self.full = True

        if new_index <= self.size:
            self._safe_add(entries)
        else:
            overflow = new_index - self.size
            self._safe_add(entries[:-overflow])
            self._safe_add(entries[-overflow:])

    def update(self, indices, errors):
        priorities = errors**self.alpha + self.min_prob
        self.priorities[indices] = priorities

    def is_full(self):
        return self.full
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.data_index
        
    def num_new_data(self):
        return len(self.new_data)
        
    def __getitem__(self, index):
        if not (0 <= index < self.__len__()):
            raise IndexError(f"Index {index} is out of bounds for size {self.size}")

        return self.data[index]