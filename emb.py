#import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from optimum.intel import OVModelForFeatureExtraction

import faiss


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
#model = AutoModel.from_pretrained("thenlper/gte-small")
model = OVModelForFeatureExtraction.from_pretrained("thenlper/gte-small",export=True)


@torch.no_grad()
def make_embedding(texts):
    # Tokenize the input texts
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    #print(outputs.keys())
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings.cpu().numpy()

class VectorDB:
    def __init__(self):
        self.texts=[]
        self.index=None

    def add(self,texts: list):
        self.texts+=texts
        embeddings=make_embedding(texts)

        if(self.index==None):
            self.index=faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)

    def search(self,texts: list,k: int,add: bool):
        if self.index==None:
            if add:
                self.add(texts)
            return [[] for _ in texts]

        embeddings=make_embedding(texts)
        distances, indices = self.index.search(embeddings, k)
        if add:
            self.texts+=texts
            self.index.add(embeddings)
        
        print(indices)
        return [[self.texts[i] for i in t] for t in indices]


#def add_user_message(index,)

if __name__=="__main__":
    print(model)
    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms"
    ]

    query_texts = [
        "python java c++"
    ]
    # embeddings=make_embedding(input_texts)
    
    # scores = (embeddings[:1] @ embeddings[1:].T) * 100
    # print(scores.tolist())
    db=VectorDB()
    db.add(input_texts)
    print(db.search(query_texts,1,True))
    print(db.texts)

    db=VectorDB()
    print(db.search(['text'],k=3,add=True))