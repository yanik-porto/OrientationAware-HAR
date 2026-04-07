import json

from encoder.encoders import *

def filter_texts_with_map(texts, classes_map, num_classes):
    if classes_map is None:
        return texts[:num_classes]

    new_texts = [""] * num_classes
    for key, val in classes_map.items():
        new_texts[val] = texts[key]
    return new_texts

class Language:
    def __init__(self, config, dl_train, dl_val, text_encoder=None):
        assert 'language' in config
        cfg_lg = config['language']

        self.mode = cfg_lg['mode']
        assert self.mode in ('labels', 'json', 'embeds')

        self.text_encoder = text_encoder
        if self.text_encoder == None or self.mode == "embeds":
            assert self.mode =='embeds'
            # TODO : check if text_embeds class mapping should be done here
            if dl_train is not None:
                assert dl_train.dataset.text_embeds is not None
                self.text_embeds_train = dl_train.dataset.text_embeds
            if dl_val is not None:
                assert dl_val.dataset.text_embeds is not None
                self.text_embeds_val = dl_val.dataset.text_embeds
            pass

        if self.mode =="labels":
            if dl_train is not None:
                self.texts_train = dl_train.dataset.label_map
            if dl_val is not None:
                self.texts_val = dl_val.dataset.label_map

        if self.mode == 'json':
            with open(cfg_lg['json_file'], 'r') as f:
                textmap = json.load(f)
            
            if dl_train is not None:
                self.a2v_train = self.load_action2viewsdesc(textmap, dl_train)
                self.indexes_keyword_train = cfg_lg['indexes_keyword_train']
            if dl_val is not None:
                self.a2v_val = self.load_action2viewsdesc(textmap, dl_val)
                self.indexes_keyword_test = cfg_lg['indexes_keyword_test']



    def load_action2viewsdesc(self, textmap, dl):
        action2viewsdesc = []
        for label_vals in textmap.values():
            viewlist = []
            for view_vals in label_vals.values():
                viewlist.append(view_vals)
            action2viewsdesc.append(viewlist)
        action2viewsdesc = filter_texts_with_map(action2viewsdesc, dl.dataset.classes_map, dl.dataset.num_classes)
        return action2viewsdesc

    def get_feats_train(self, labels, batch):
        if self.mode == 'embeds':
            return torch.from_numpy(np.array([self.text_embeds_train[label] for label in labels], dtype=np.float32))
        
        elif self.mode == 'labels':
            texts = [self.texts_train[label] for label in labels]
            return self.text_encoder(texts)

        elif self.mode == 'json':
            indexes = self.select_indexes_from_batch(batch, self.indexes_keyword_train)
            texts = self.get_texts_train(labels, indexes)
            return self.text_encoder(texts)

        else:
            raise NotImplementedError()

    def select_indexes_from_batch(self, batch, indexes_keyword):
        if indexes_keyword.isdigit():
            return [int(indexes_keyword)] * len(batch['keypoint'])
        elif indexes_keyword == 'random':
            return np.random.randint(0, len(self.a2v_train[0]), size=len(batch['keypoint'])).tolist()

        assert indexes_keyword in  batch, str(batch.keys()) + "vs " + indexes_keyword
        return batch[indexes_keyword][0] #for now, as long as no multiview training with specified index

    def select_indexes_from_batch_test(self, batch):
        if self.indexes_keyword_test == 'random':
            raise NotImplementedError
        return self.select_indexes_from_batch(batch, self.indexes_keyword_test)
    
    def get_texts_train(self, labels, indexes):
        texts = []
        for label, idx in zip(labels, indexes):
            viewlist = self.a2v_train[label]
            texts.append(viewlist[idx])

        return texts

    def get_feats_test(self, batch=None):
        if self.mode == 'embeds':
            return torch.from_numpy(self.text_embeds_val)
        else:
            texts = self.get_texts_test(batch)

            # check if list of list of text or list of text
            if len(texts) > 0 and type(texts[0]) is list:
                embedds = []
                for text_list in texts:
                   embedds.append(self.text_encoder(text_list))
                embedds = torch.stack(embedds, dim=0)
                return embedds
            else:
                return self.text_encoder(texts)
        
    def get_texts_test(self, batch=None):
        if self.mode == "labels":
            return self.texts_val
        elif self.indexes_keyword_test == "all":
            return self.get_texts_from_all_views()
        elif self.indexes_keyword_test.isdigit():
            idx = int(self.indexes_keyword_test)
            return self.get_texts_from_view(idx)
        else:
            assert batch is not None
            indexes = self.select_indexes_from_batch_test(batch)
            return self.get_texts_from_view(indexes)

    def get_texts_from_view(self, index):
        texts = []
        for viewlist in self.a2v_val:
            assert index < len(viewlist), str(len(viewlist)) + " vs " + str(index)
            texts.append(viewlist[index])
        return texts

    def get_texts_from_all_views(self):
        return [list(row) for row in zip(*self.a2v_val)]
        
        


        