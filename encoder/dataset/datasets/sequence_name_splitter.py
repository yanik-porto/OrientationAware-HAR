pkummd_cam_map = {'L': 0, 'M': 1, 'R': 2}

class SequenceNameSplitter():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        if ("_tad" in self.dataset_name and self.dataset_name != "pkummd_tad") or self.dataset_name == "tsu":
            self.dataset_name = "ntu_tad"
            print(f"Dataset name {self.dataset_name} not explicitly handled. Considering default dataset {self.dataset_name} for naming convention")

        if self.dataset_name not in ("babel_mv", "babel", "ntu", "ntu_tad", "pkummd_tad"):
            self.dataset_name = "ntu"
            print(f"Dataset name {self.dataset_name} not explicitly handled. Considering default dataset {self.dataset_name} for naming convention")


    def get_group_from_name(self, seq_name):
        splits = seq_name.split('_')

        motionid_with_action = ""
        camera_id = ""
        if self.dataset_name == "babel_mv":
            motionid_with_action, camera_id = splits[0] + '_' + splits[1] + '_' + splits[3], int(splits[2].split('Camera')[1])
        elif self.dataset_name == "babel":
            motionid_with_action, camera_id = splits[0] + '_' + splits[1] + '_' + splits[2], 0
        elif self.dataset_name == "ntu":
            motionid_with_action, camera_id = seq_name[0:4] + seq_name[8:20], int(seq_name[4:8][1:])#-1 # TODO : check if -1 correct for NTU
        elif self.dataset_name == "ntu_tad":
            assert len(seq_name) > 10
            motionid_with_action, camera_id = seq_name[0:4] + seq_name[8:20], int(seq_name[4:8][1:])
        elif self.dataset_name == "pkummd_tad":
            assert len(seq_name) <= 10
            motionid_with_action = seq_name[0:4]
            camera_id = pkummd_cam_map[seq_name[5]]
        else:
            raise NotImplementedError()

        return motionid_with_action, camera_id
    
    def get_motionid_from_name(self, seq_name):
        splits = seq_name.split('_')

        motionid = ""
        camera_id = ""
        if self.dataset_name == "babel_mv":
            assert len(splits) > 2, splits
            assert 'Camera' in splits[2], splits
            motionid, camera_id = splits[0] + '_' + splits[1], int(splits[2].split('Camera')[1])
        elif self.dataset_name == "babel":
            assert len(splits) > 2, splits
            motionid, camera_id = splits[0] + '_' + splits[1], 0
        elif self.dataset_name == "ntu":
            motionid, camera_id = self.get_group_from_name(seq_name)
        elif self.dataset_name in ("ntu_tad", "pkummd_tad"):
            motionid, camera_id = self.get_group_from_name(seq_name)
        else:
            raise NotImplementedError()

        return motionid, camera_id