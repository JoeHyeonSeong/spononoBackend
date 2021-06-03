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


app=Flask(__name__)
@app.route('/predict',methods=["POST"])
def predict():
    start=datetime.datetime.now()
    received_data=request.json
    text=received_data['contents']
    #check if exist in DB
    dbResult=db.texts.find({"text":text})
    result=False
    hit=False
    #cache hit
    if(dbResult.count()>0):
        result=dbResult[0]["spoiler"]
        hit=True
    else:#cache miss
        hit=False
        transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=max_len,pad= True, pair=False)
        transText=transform([text])
        token_id=torch.tensor([transText[0]])
        valid_length=torch.tensor(np.array([transText[1]]))
        seq_id=torch.tensor([transText[2]])
        result=model(token_id.long(),valid_length,seq_id.long())
        max_vals, max_indices = torch.max(result, 1)
        result=max_indices[0].item()==1
        try:
            db.texts.insert({"text":text,"spoiler":result})
        except:
            print("db insert failed")
    elapsed=(datetime.datetime.now()-start).total_seconds()
    db.predictLog.insert({"date":datetime.datetime.now(),"hit":hit,"responseTime":elapsed,"isSpoiler":result})
    print("\n\nPREDICT\n",text+"\n","result: "+str(result)+" hit: "+str(hit)+" response time: "+str(elapsed)+"\n")
    return jsonify(output=result)

@app.route('/swearPredict',methods=["POST"])
def swearPredict():
    start=datetime.datetime.now()
    received_data=request.json
    text=received_data['contents']
    #check if exist in DB
    dbResult=db.swearTexts.find({"text":text})
    result=False
    hit=False
    #cache hit
    if(dbResult.count()>0):
        result=dbResult[0]["swear"]
        hit=True
    else:#cache miss
        hit=False
        transform = nlp.data.BERTSentenceTransform(swTok, max_seq_length=max_len,pad= True, pair=False)
        transText=transform([text])
        token_id=torch.tensor([transText[0]])
        valid_length=torch.tensor(np.array([transText[1]]))
        seq_id=torch.tensor([transText[2]])
        result=swModel(token_id.long(),valid_length,seq_id.long())
        max_vals, max_indices = torch.max(result, 1)
        result=max_indices[0].item()==1
        try:
            db.swearTexts.insert({"text":text,"swear":result})
        except:
            print("db insert failed")
    elapsed=(datetime.datetime.now()-start).total_seconds()
    print("\n\nSWEAR PREDICT\n",text+"\n","result: "+str(result)+" hit: "+str(hit)+" response time: "+str(elapsed)+"\n")
    return jsonify(output=result)


@app.route('/report',methods=["POST"])
def report():
    received_data=request.json
    text=received_data["text"]
    spoiler=received_data["isSpoiler"]
    print("REPORT\n",text)

    try:
        dbResult=db.reported.find_one({"text":text})
        spoilerAdd=0
        noSpoilerAdd=0
        if(spoiler):
            spoilerAdd+=1
        else:
            noSpoilerAdd+=1
        if(dbResult):
            spoilerAdd+=dbResult["spoiler"]
            noSpoilerAdd+=dbResult["noSpoiler"]
            db.reported.update_one({"text":text},{"$set":{"spoiler":spoilerAdd,"noSpoiler":noSpoilerAdd}})
        else:
            db.reported.insert({"text":text,"spoiler":spoilerAdd,"noSpoiler":noSpoilerAdd})
        
        isSpoiler=(spoilerAdd>=noSpoilerAdd)
        textDB=db.texts.find_one({"text":text})
        if(textDB):
            db.texts.update_one({"text":text},{"$set":{"spoiler":isSpoiler}})
        else:
            db.texts.insert({"text":text,"spoiler":isSpoiler})
    except:
        print("db insert failed")
    
    return {}


@app.route('/search',methods=['GET'])
def search():
    received_data=request.args.to_dict()
    title=quote(received_data['title'])
    movieType=quote("극장용")
    basicUrl = "http://api.koreafilm.or.kr/openapi-data2/wisenut/search_api/search_json2.jsp?" \
            "collection=kmdb_new2&ServiceKey=M9RA61A20074QJD5W74X&use="+movieType+"&sort=prodYear,1&detail=Y&listCount=500&title="
    req=urlRequest.Request(basicUrl+title)
    res = urlRequest.urlopen(req)
    if(res.status==200):
        return res.read().decode('utf-8')
    else:
        return {}

def make_model(modelfile, vocabfile):
    print("loading model...")
    from sponono import BERTClassifier
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
port=5000
#os.environ["TOKENIZERS_PARALLELISM"] = "false"
db=connectDB()
model,tok=make_model("classifier.pt","vocab.vocab")
swModel,swTok=make_model("swearClassifier.pt","swearvocab.vocab")

if(__name__=='__main__'):
    print("start!!!")
    app.run(host='0.0.0.0',port=port)

