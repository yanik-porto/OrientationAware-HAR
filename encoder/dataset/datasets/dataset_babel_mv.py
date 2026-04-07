from .dataset_mv import DatasetMV

import pickle
import sys
import os.path as op
sys.path.insert(0, op.join(op.dirname(__file__), '../'))
import torch

class DatasetBabelMV(DatasetMV):
    def __init__(self, data_path, split='xsub_train', n_views=2, num_classes=120, preprocessing=False, classes_map=None, label_map=None, oneshot=False, motionid_labels_path="", **kwargs):
        self.motionid_to_labels = self.load_motionid_to_labels(motionid_labels_path, split)
        super().__init__(data_path, "babel_mv", split, n_views, num_classes, classes_map, preprocessing, label_map, oneshot, **kwargs)

    def load_motionid_to_labels(self, path, split):
        if path == "":
            return None
        abs_path = op.join(op.dirname(__file__), path)
        with open(abs_path, 'rb') as f:
            data = pickle.load(f)
        if data is None or split not in data:
            return None

        return data[split]
    
    def save_binarylabels(self, data):
        if self.motionid_to_labels == None:
            super().save_binarylabels(data)
            return

        for idx in range(len(data)):
            motionid, _ = self.get_motionid_from_name(data[idx]["frame_dir"])
            mlabels = self.motionid_to_labels[motionid]
            m_binary_labels = torch.zeros(self.num_classes, dtype=torch.uint8)
            for mlabel in mlabels:
                if mlabel >= self.num_classes:
                    continue
                m_binary_labels[mlabel] = 1
            data[idx]['binary_labels'] = m_binary_labels