from pymongo import MongoClient
def connectDB():
    client=MongoClient('mongodb://localhost:27017/')
    db=client.sponono
    return db

def test():
    spoilerSampleNum=1
    noSpoilerSampleNum=1
    spoilerRealNum=1
    noSpoilerRealNum=2
    #get data for prove
    spoilerSample=db.sample.aggregate([
        {"$match":{"isSpoiler":True}},
        {"$sample":{"size":spoilerSampleNum}}
        ])

    noSpoilerSample=db.sample.aggregate([
        {"$match":{"isSpoiler":False}},
        {"$sample":{"size":noSpoilerSampleNum}}
        ])
    sample=[]
    sample+=spoilerSample
    sample+=noSpoilerSample

    question=[]
    texts=db.texts.aggregate([
        {"$lookup":{
            "from":"reported",
            "localField":"text",
            "foreignField":"text",
            "as":"reported"
            }
        }
    ])
    spoilerCnt=0
    noSpoilerCnt=0
    questions=[]
    for t in texts:
        if(t["reported"]==[]):
            if(t["spoiler"]):
                if(spoilerCnt<spoilerRealNum):
                    spoilerCnt+=1
                    questions.append(t)
            else:
                if(noSpoilerCnt<noSpoilerRealNum):
                    noSpoilerCnt+=1
                    questions.append(t)
        if(spoilerCnt==spoilerRealNum and noSpoilerCnt==noSpoilerRealNum):
            break
    result={"samples":sample,"questions":questions}
    
    print(result)
    return result

db=connectDB()
test()
