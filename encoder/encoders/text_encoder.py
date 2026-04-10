import torch
import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class TextEncoder():
    def __init__(self, wpath='distilbert-base-uncased', loader='automodel', split_coma=False, n_splits=3):#, do_normalize=False):
        assert loader in ('automodel', 'sentence_transformer')
        self.loader = loader
        if self.loader == 'automodel':
            self.model = AutoModel.from_pretrained(wpath)
            self.tokenizer = AutoTokenizer.from_pretrained(wpath, TOKENIZERS_PARALLELISM=True)
            self.target_token_idx = 0
        elif self.loader == 'sentence_transformer':
            self.model = SentenceTransformer(wpath)

        self.split_coma = split_coma
        self.n_splits = n_splits

        self.eval()

    def to(self, device):
        self.device = device
        self.model = self.model.to(self.device)

    def forward(self, texts):
        embeddings = None

        if self.split_coma:
            N = len(texts)
            texts_split = []
            for text in texts:
                text_split = text.split(', ')
                assert len(text_split) >= self.n_splits, text_split
                if len(text_split) > self.n_splits:
                    if self.n_splits > 2:
                        text_split = text_split[0:2] + text_split[-1:]
                    else:
                        text_split = text_split[0:self.n_splits]
                texts_split += text_split
            texts = texts_split
            assert len(texts) == self.n_splits*N

        if self.loader == 'automodel':
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            output = self.model(input_ids=tokens['input_ids'], attention_mask=tokens["attention_mask"])
            last_hidden_state = output.last_hidden_state
            embeddings =  last_hidden_state[:, self.target_token_idx, :]#.detach().cpu().numpy()
        elif self.loader == 'sentence_transformer':
            embeddings = torch.tensor(self.model.encode(texts)).to(self.device)


        if self.split_coma:
            embedding_merged = embeddings.reshape(N, self.n_splits, -1)
            embeddings = embedding_merged.permute(0, 2, 1)
            embeddings = embeddings.contiguous()
            embeddings = embeddings.view(N, -1)

        return embeddings
    
    def __call__(self, texts):
        return self.forward(texts)
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()