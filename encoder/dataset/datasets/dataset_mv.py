from torch.utils.data import Dataset
import pickle
import os
import os.path as op
import sys
import os.path as osp
import torch
from abc import abstractmethod
import copy
import numpy as np
import random

SEP = " | "

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from preprocessors import *
from .sequence_name_splitter import SequenceNameSplitter

class DatasetMV(Dataset):
    def __init__(self, data_path, dataset_name, split='xsub_train', n_views=2, num_classes=120, classes_map=None, preprocessing=False,
                 label_map=None, oneshot=False, max_samples_per_class=5000, respect_cams_order=True, closest_node_only=-1,
                 text_embeds_suffix='_clip.npy', is_dynamic_loading=False):
        assert os.path.exists(data_path), data_path + " not found"

        self.names_splitter = SequenceNameSplitter(dataset_name)
        self.split = split
        self.n_views = n_views
        self.num_classes = num_classes
        self.classes_map = classes_map
        self.preprocessing = preprocessing
        self.oneshot = oneshot
        self.max_samples_per_class = max_samples_per_class
        self.respect_cams_order = respect_cams_order
        self.closest_node_only = closest_node_only
        self.label_map = [x.strip() for x in open(op.join(op.dirname(__file__), label_map)).readlines()] if label_map is not None else []
        self.text_embeds = self.load_text_embeds(label_map, text_embeds_suffix)
        self.label_map = self.map_labels_map(self.label_map)# if self.classes_map is not None else self.label_map


        #self.map_assocs_samples = {0: [1, 2, 3], 1: [2, 3, 0], 2: [0, 1, 3], 3: [0, 1, 2]}
        # self.map_assocs_samples = {1: [2, 3], 2: [3, 1], 3: [1, 2]}
        self.map_assocs_samples = {1: [2, 3, 0], 2: [3, 1, 0], 3: [1, 2, 0], 0: [1, 2, 3]}
        # self.map_assocs_samples = {0: [1, 2, 3]}

        if not is_dynamic_loading:
            self.load_data(data_path)


    def load_data(self, data_path):
        assert os.path.exists(data_path), data_path + " not found"
        with open(data_path, 'rb') as d: data = pickle.load(d)
        data = self.filter_annotations(data)
        self.save_binarylabels(data)
        self.data, self.map_idx_ok = self.find_associated_ids(data)

        if self.oneshot:
            self.map_idx_ok = self.keep_one_per_class(self.map_idx_ok)


    def save_binarylabels(self, data):
        for idx in range(len(data)):
            if 'binary_labels' in data[idx]:
                continue
            m_binary_labels = torch.zeros(self.num_classes, dtype=torch.uint8)
            m_binary_labels[data[idx]['label']] = 1
            data[idx]['binary_labels'] = m_binary_labels

    def keep_one_per_class(self, mapidx):
        idx_kepts = []
        labels_found = []
        for idx in mapidx:
            assert idx < len(self.data), str(idx) + " >= " + len(self.data)
            d = self.data[idx]
            if d["label"] in labels_found:
                continue
            idx_kepts.append(idx)
            labels_found.append(d["label"])
        return idx_kepts

    def filter_with_classes_map(self, data):
        # keep only classes in the map and change by corresponding classes
        if self.classes_map is not None:
            data_filtered = []
            for d in data:
                if d['label'] in self.classes_map:
                    d['label'] = self.classes_map[d['label']]
                    data_filtered.append(d)
            data = data_filtered

        # check number of samples in each class, to equilibrate
        n_per_class = {}
        data_filtered = []
        for d in data:
            l = d['label']
            if not l in n_per_class:
                n_per_class[l] = 0

            if n_per_class[l] >= self.max_samples_per_class:
                continue

            data_filtered.append(d)
            n_per_class[l] += 1
        data = data_filtered

        return data

    def filter_annotations(self, data):
        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data =  [x for x in data if x[identifier] in split]
            data = self.filter_with_classes_map(data)
            data = [x for x in data if x['label'] < self.num_classes]

            if self.closest_node_only >= 0:
                data = [x for x in data if x['closest_node'][0] == self.closest_node_only]
        return data
    
    def update_if_needed(self, losses=None):
        self.preprocessing.update()
        if losses is not None:
            self.preprocessing.update_with_losses(losses)

    def find_associated_ids(self, data):
        map_index_with_assocs = []

        if self.n_views == 1:
            map_index_with_assocs = list(range(len(data)))
            if self.text_embeds is not None:
                 for d in data: d["text_embed"] = self.text_embeds[d["label"]]
            return data, map_index_with_assocs

        for idx in range(len(data)):
            assocs = self.get_associated_ids(idx, data)

            if len(assocs) + 1 < self.n_views:
                print("index #", idx, " (with  name ", data[idx]["frame_dir"], " belongs to a group with too few views : ", len(assocs) + 1, " < ", self.n_views)
                continue
            
            data[idx]["assocs"] = assocs

            if self.text_embeds is not None:
                data[idx]["text_embed"] = self.text_embeds[data[idx]["label"]]

            map_index_with_assocs.append(idx)

        return data, map_index_with_assocs
    
    def get_group_from_name(self, name):
        return self.names_splitter.get_group_from_name(name)

    def get_motionid_from_name(self, name):
        return self.names_splitter.get_motionid_from_name(name)


    def get_associated_ids(self, idx, data):
        assocs = []

        group, camCurrent = self.get_group_from_name(data[idx]["frame_dir"])
        if camCurrent not in self.map_assocs_samples:
            return assocs
        mapAssocs = self.map_assocs_samples[camCurrent]

        for idxOther, other  in enumerate(data):
            if idxOther == idx:
                continue
            groupOther, camOther = self.get_group_from_name(other["frame_dir"])
            if groupOther == group:
                # check if associated cam is close enough in the cam order
                assert camOther in mapAssocs, f"{camOther} vs {mapAssocs} in ({group}, {camCurrent}, {idx}) and ({groupOther}, {camOther}, {idxOther})" 
                idxInAssocs = mapAssocs.index(camOther) + 1 # + 1 for current camera
                if self.respect_cams_order:
                    if idxInAssocs >= self.n_views:
                        continue
                assocs.append((camOther, idxOther))
        return assocs
    
    def __len__(self):
        """Get the size of the dataset."""
        return len(self.map_idx_ok)

    def merge_annot_into_other(self, annot, other):
        if False and other['keypoint'].shape[1] != annot['keypoint'].shape[1]:
            print("watchout, temporal dimension mismatch : ", other['keypoint'].shape[1], " vs ", annot['keypoint_score'].shape[1])

        concat = copy.copy(other)

        assert 'keypoint' in concat, str(concat.keys())
        assert 'keypoint' in annot, str(annot.keys())

        # assert concat[keypoint_str].shape[0] == 1, str(concat[keypoint_str].shape[0])
        # assert annot[keypoint_str].shape[0] == 1, str(annot[keypoint_str].shape[0])

        T = min(concat['keypoint'].shape[1], annot['keypoint'].shape[1])
        concat['total_frames'] = T
        concat['keypoint'] = np.concatenate((concat['keypoint'][:, :T, :, :], annot['keypoint'][:, :T, :, :]))
        
        if 'keypoint_score' in annot and 'keypoint_score' in concat:
            concat['keypoint_score'] = np.concatenate((concat['keypoint_score'][:, :T, :], annot['keypoint_score'][:, :T, :]))
        
        if 'orientation' in annot and 'orientation' in concat:
            concat['orientation'] = np.concatenate((concat['orientation'], annot['orientation']))

        if 'indexes' in annot and 'indexes' in concat:
            concat['indexes'] = np.concatenate((concat['indexes'], annot['indexes']))

        return concat

    def merge_samples(self, samples):
        origin = samples[0] # TODO : check if copy needed
        for i in range(1, len(samples)):
            origin = self.merge_annot_into_other(samples[i], origin)
        return origin

    def __getitem__(self, idx_required):
        """Get the sample for either training or testing given index."""

        assert(len(self.map_idx_ok) > idx_required)

        idx_mapped = self.map_idx_ok[idx_required]
        
        d = copy.deepcopy(self.data[idx_mapped])

        samples = [dict()] * self.n_views
        samples[0] = self.preprocessing(d)

        if self.n_views > 1:
            _, camCurrent = self.get_group_from_name(d["frame_dir"])
            for id, (camAssoc, idx_assoc) in enumerate(d["assocs"]):
                if self.respect_cams_order:
                    mapAssocs = self.map_assocs_samples[camCurrent]
                    if camAssoc not in mapAssocs:
                        continue
                    idxInSamples = mapAssocs.index(camAssoc) + 1 # + 1 for current camera
                else:
                    idxInSamples = id + 1
                if idxInSamples >= self.n_views:
                    continue

                ass = copy.deepcopy(self.data[idx_assoc])
                samples[idxInSamples] = self.preprocessing(ass)

        for s in samples:
            assert len(s.keys()) > 0, str(camCurrent) + " : " + str(d["assocs"])

        return self.merge_samples(samples)

    def load_text_embeds(self, label_map_path, text_embeds_suffix):
        te_path = op.splitext(label_map_path)[0] + text_embeds_suffix
        # te_path = op.splitext(label_map_path)[0] + '_distilbert.npy'
        te_path = op.join(op.dirname(__file__), te_path)
        if op.isfile(te_path):
            text_embeds = np.load(te_path)
            if len(self.label_map) != len(text_embeds):
                print("text embeddings length mismatch labels : ", len(text_embeds), " vs ", len(self.label_map))
                return None
            
            if self.classes_map is not None:
                text_embeds_mapped = np.zeros(text_embeds.shape, dtype=text_embeds.dtype)
                for key, val in self.classes_map.items():
                    text_embeds_mapped[val] = text_embeds[key]
                text_embeds = text_embeds_mapped
            text_embeds = text_embeds[:self.num_classes]
            return text_embeds
        return None
    
    def map_labels_map(self, label_map):
        if self.classes_map is None:
            return label_map[:self.num_classes]

        new_map = [""] * self.num_classes
        for key, val in self.classes_map.items():
            new_map[val] = label_map[key]
        return new_map
    
    def label_to_text(self, label, training=False):
        text = self.label_map[label]
        splits = text.split(SEP)
        assert len(splits) > 0, text
        if len(splits) < 2:
            return text
        else:
            i = random.randrange(len(splits)) if training else 0
            return splits[i]
            
