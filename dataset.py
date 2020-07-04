"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors, 
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from transforms import color_aug


GOP_SIZE = 12


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class GoPDataSet(data.Dataset):
    def __init__(self, 
                data_root, 
                video_list,
                transform,
                num_segments,
                is_train,
                ):

        self._data_root = data_root
        self._num_segments = 3
        self._transform = transform #TODO:
        self._is_train = is_train #TODO: TESTING DATASET
        self._accumulate = False

        self._rgb_input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._rgb_input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r') as f:
            for line in f:
                video, label, frames = line.strip().split()
                video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                self._video_list.append((video_path, int(label), int(frames)))

        print('%d videos loaded.' % len(self._video_list))


    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                                 representation='mv')

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, 'mv')

    def _get_test_frame_index(self, num_frames, seg):
        num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        v_frame_idx += 1

        return get_gop_pos(v_frame_idx, 'mv')

    def __getitem__(self, index):

        repre_idx_mv = 1
        repre_idx_res = 2
        repre_idx_rgb = 0

        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        i_frame_gathered = []
        mv_gathered = []
        res_gathered = []
        for seg in range(self._num_segments):

            if self._is_train:
                gop_index, _ = self._get_train_frame_index(num_frames, seg)
            else:
                gop_index, _ = self._get_test_frame_index(num_frames, seg)

            iframe_img = load(video_path, gop_index, 0, repre_idx_rgb, self._accumulate)
            if iframe_img is None:
                print('Error: loading video %s failed.' % video_path)
                iframe_img = np.zeros((256, 256, 3))
            iframe_img = color_aug(iframe_img)
            # BGR to RGB. (PyTorch uses RGB according to doc.)
            iframe_img = iframe_img[..., ::-1]
            i_frame_gathered.append(iframe_img) #lens = segs


            for gop_pos in range(GOP_SIZE-1):
                mv_img = load(video_path, gop_index, gop_pos, repre_idx_mv, self._accumulate)
                if mv_img is None:
                    print('Error: loading video %s failed.' % video_path)
                    mv_img = np.zeros((256, 256, 2)) 
                mv_img = clip_and_scale(mv_img, 20)
                mv_img += 128
                mv_img = (np.minimum(np.maximum(mv_img, 0), 255)).astype(np.uint8)

                res_img = load(video_path, gop_index, gop_pos, repre_idx_res, self._accumulate)
                if res_img is None:
                    print('Error: loading video %s failed.' % video_path)
                    res_img = np.zeros((256, 256, 3))
                res_img += 128
                res_img = (np.minimum(np.maximum(res_img, 0), 255)).astype(np.uint8)

                mv_gathered.append(mv_img)  #lens = segs*(GOP_SIZE-1)
                mv_gathered.append(res_img) #lens = segs*(GOP_SIZE-1)
            

        #TODO: currently rgb transform is not the same with rgb and residual, need to coordinate their transform to be the same in 
        #one single GOP
        i_frame_gathered, mv_gathered, res_gathered = self._transform(i_frame_gathered, mv_gathered, res_gathered)

        i_frame_gathered = np.array(i_frame_gathered)
        i_frame_gathered = np.transpose(i_frame_gathered, (0, 3, 1, 2))
        i_frame_gathered_input = torch.from_numpy(i_frame_gathered).float() / 255.0

        mv_gathered = np.array(mv_gathered)
        mv_gathered = np.transpose(mv_gathered, (0, 3, 1, 2))
        mv_gathered_input = torch.from_numpy(mv_gathered).float() / 255.0
        mv_gathered_input = mv_gathered_input.view((-1, self._num_segments) + mv_gathered_input.size()[1:])

        res_gathered = np.array(res_gathered)
        res_gathered = np.transpose(res_gathered, (0, 3, 1, 2))
        res_gathered_input = torch.from_numpy(res_gathered).float() / 255.0
        res_gathered_input = res_gathered_input.view((-1, self._num_segments) + res_gathered_input.size()[1:])

        i_frame_gathered_input = (i_frame_gathered_input - self._rgb_input_mean) / self._rgb_input_std
        mv_gathered_input = (mv_gathered_input - 0.5)
        res_gathered_input = (res_gathered_input - 0.5) / self._rgb_input_std

        return i_frame_gathered_input, mv_gathered_input, res_gathered_input, label

    def __len__(self):
        return len(self._video_list)
