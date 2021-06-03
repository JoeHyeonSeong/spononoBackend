import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from flask import Flask,request,jsonify
from kobert.utils import get_tokenizer
import gluonnlp as nlp
import numpy as np
import os
from pymongo import MongoClient
from urllib import request as urlRequest
from urllib.parse import quote
import datetime


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

def update():
    cache=db.texts.find()
    for c in cache:
        text=c["text"]
        transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len,pad= True, pair=False)
        transText=transform([text])
        token_id=torch.tensor([transText[0]])
        valid_length=torch.tensor(np.array([transText[1]]))
        seq_id=torch.tensor([transText[2]])
        result=model(token_id.long(),valid_length,seq_id.long())
        max_vals, max_indices = torch.max(result, 1)
        result=max_indices[0].item()==1
        print(text)
        print(c["spoiler"])
        print(result)
        print("\n")
        db.texts.update_one({"text":text},{"$set":{"spoiler":result}})


def make_model(modelfile, vocabfile):
    print("loading model...")
    model=torch.load(modelfile,map_location=torch.device('cpu'))
    model.eval()
    vocab=torch.load(vocabfile)
    tokenizer=get_tokenizer()
    tok= nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    return model,tok


def connectDB():
    client=MongoClient('mongodb://localhost:27017/')
    db=client.sponono
    return db

max_len=64
batch_size = 64
db=connectDB()
directory="/home/joo9245/spononoBackend/"
model,tok=make_model(directory+"classifier.pt",directory+"vocab.vocab")
update()
