# -------------------------------------------------------------------------
# Copyright (c) 2021 Jie Lei
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Source: https://github.com/microsoft/xpretrain/tree/main/CLIP-ViP
# This file is original from the CLIP-ViP project.

"""
modified from UNITER codebase

A meta data loader for sampling from different datasets / training tasks
A prefetch loader to speedup data loading
"""
import random

import torch
from torchvision import transforms
from torch.utils.data import DataLoader



class MetaLoader(object):
    """ wraps multiple data loader """
    def __init__(self, loaders, accum_steps=1, distributed=False, epoch=2):
        assert isinstance(loaders, dict)
        self.name2loader = {}
        self.name2iter = {}
        self.name2epoch = {}
        self.sampling_pools = []
        n_batches_in_epoch = 0
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            n_batches_in_epoch += len(l.dataset) * r / l.batch_size
            self.name2loader[n] = l
            self.name2epoch[n] = epoch
            self.name2loader[n].sampler.set_epoch(self.name2epoch[n])
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n]*r)
        self.n_batches_in_epoch = n_batches_in_epoch
        self.accum_steps = accum_steps
        self.distributed = distributed
        self.step = 0

    def __iter__(self):
        """ this iterator will run indefinitely """
        task = self.sampling_pools[0]
        while True:
            if self.step % self.accum_steps == 0:
                task = random.choice(self.sampling_pools)
            self.step += 1
            iter_ = self.name2iter[task]
            try:
                batch = next(iter_)
            except StopIteration:
                self.name2epoch[task] += 1
                self.name2loader[task].sampler.set_epoch(self.name2epoch[task])
                iter_ = iter(self.name2loader[task])
                batch = next(iter_)
                self.name2iter[task] = iter_

            yield task, batch


def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass


class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """
    def __init__(self, loader, img_normalize=None):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.img_normalize = img_normalize

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            is_tuple = isinstance(batch, tuple)
            if is_tuple:
                task, batch = batch
            # batch["images"] = batch["images"].float()
            # if self.img_normalize is not None:
            #     batch["images"] = self.img_normalize(batch["images"])
            if is_tuple:
                yield task, batch
            else:
                yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


class InfiniteIterator(object):
    """iterate an iterable oobject infinitely"""
    def __init__(self, iterable):
        self.iterable = iterable
        self.epoch = 0
        # Only set the epoch if the sampler has the `set_epoch` method
        if hasattr(self.iterable.sampler, 'set_epoch'):
            self.iterable.sampler.set_epoch(self.epoch)
        self.iterator = iter(iterable)

    def __iter__(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.epoch += 1
                if hasattr(self.iterable.sampler, 'set_epoch'):
                    self.iterable.sampler.set_epoch(self.epoch)
                self.iterator = iter(self.iterable)
                batch = next(self.iterator)
            yield batch


def init_transform_dict(video_res=(240, 320),
                        input_res=(224, 224),
                        randcrop_scale=(0.4, 0.8),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize([video_res[0], video_res[1]], antialias=True),
            transforms.CenterCrop([int(video_res[0]*0.6), int(video_res[1]*0.6)]),
            transforms.Resize(input_res, antialias=True),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize([video_res[0], video_res[1]], antialias=True),
            transforms.CenterCrop([int(video_res[0]*0.6), int(video_res[1]*0.6)]),
            transforms.Resize(input_res, antialias=True),
            normalize,
        ])
    }
    return transform_dict

def init_transform_dict_simple(video_res=(240, 320),
                        input_res=(224, 224),
                        randcrop_scale=(0.8, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform_dict = {
        'train': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ])
    }
    return transform_dict

def init_transform_dict_image(video_res=(240, 320),
                        input_res=(224, 224),
                        randcrop_scale=(0.8, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.48145466, 0.4578275, 0.40821073),
                        norm_std=(0.26862954, 0.26130258, 0.27577711)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_res, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(input_res),
            normalize,
        ])
    }
    return transform_dict