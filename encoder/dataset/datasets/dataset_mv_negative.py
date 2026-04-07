from .dataset_mv import DatasetMV

import numpy as np
import copy
import pickle
import os.path as op
import sys
sys.path.insert(0, op.join(op.dirname(__file__), '../'))


class DatasetMVNegative(DatasetMV):
    """ in this case n_views stands for the number of different motion"""
    def __init__(self, data_path, dataset_name, split='xsub_train', n_views=2, num_classes=120, preprocessing=False, classes_map=None, label_map=None, oneshot=False, motionid_labels_path="", self_supervised=False, **kwargs):
        self.motionid_to_labels = self.load_motionid_to_labels(motionid_labels_path, split)
        np.random.seed(255)

        super().__init__(data_path, dataset_name, split, n_views, num_classes, classes_map, preprocessing, label_map, oneshot, **kwargs)

        self.self_supervised = self_supervised

    def load_motionid_to_labels(self, path, split):
        if path == "":
            return None
        abs_path = op.join(op.dirname(__file__), path)
        with open(abs_path, 'rb') as f:
            data = pickle.load(f)
        if data is None or split not in data:
            return None

        return data[split]
    
    def update_if_needed(self, losses=None):
        DatasetMV.update_if_needed(self)
        if losses is not None:
            self.preprocessing.update_with_losses(losses)
        self.recompute_associations()
        print("*dataset updated*")

    def recompute_associations(self):
        self.data, self.map_idx_ok = self.find_associated_ids(self.data)

    def find_associated_ids(self, data):
        map_index_with_assocs = []

        if self.n_views == 1:
            map_index_with_assocs = list(range(len(data)))
            if self.text_embeds is not None:
                 for d in data: d["text_embed"] = self.text_embeds[d["label"]]
            return data, map_index_with_assocs

        # self.remainids = list(range(len(data)))

        for idx in range(len(data)):
            # if idx not in self.remainids:
            #     continue

            if self.self_supervised:
                assocs = self.get_associated_ids_random(idx, data)
            else:
                assocs = self.get_associated_ids(idx, data)

            if len(assocs) + 1 < self.n_views:
                print("index #", idx, " (with  name ", data[idx]["frame_dir"], " belongs to a group with too few views : ", len(assocs) + 1, " < ", self.n_views)
                continue
            
            data[idx]["assocs"] = assocs

            if self.text_embeds is not None:
                data[idx]["text_embed"] = self.text_embeds[data[idx]["label"]]

            map_index_with_assocs.append(idx)
            # self.remainids.remove(idx)

        return data, map_index_with_assocs

    def get_associated_ids(self, idx, data):
        assocs = []

        motion_id, cam_id = self.get_motionid_from_name(data[idx]["frame_dir"])

        while len(assocs) + 1 < self.n_views:
            # riter = np.random.randint(len(self.remainids))
            # ridx = self.remainids[riter]
            ridx = np.random.randint(len(data))
            motion_id_assoc, cam_id_other = self.get_motionid_from_name(data[ridx]["frame_dir"])
            if motion_id_assoc == motion_id or cam_id_other != cam_id:
                continue
            # self.remainids.pop(riter) # remove index from list
            assocs.append((cam_id_other, ridx))
        return assocs
    
    def get_associated_ids_random(self, idx, data):
        assocs = []

        while len(assocs) + 1 < self.n_views:
            ridx = np.random.randint(len(data))
            if ridx == idx:
                continue
            assocs.append((0, ridx))
        return assocs
    
    def __getitem__(self, idx_required):
        """Get the sample for either training or testing given index."""

        assert len(self.map_idx_ok) > idx_required, f"getitem #{idx_required} in map of length {len(self.map_idx_ok)} ({self.__len__()})"

        idx_mapped = self.map_idx_ok[idx_required]
        
        d = copy.deepcopy(self.data[idx_mapped])
        # _, camCurrent = self.get_group_from_name(d["frame_dir"])

        samples = []
        samples.append(self.preprocessing(d))

        if self.n_views > 1:
            for _, idx_assoc in d["assocs"]:
                ass = copy.deepcopy(self.data[idx_assoc])
                samples.append(self.preprocessing(ass))

        assert len(samples) == self.n_views, "samples length is %d buf should be %d".format(len(samples), self.n_views) 

        # for s in samples:
            # assert len(s.keys()) > 0, str(camCurrent) + " : " + str(d["assocs"])

        return self.merge_samples(samples)