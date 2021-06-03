from pymongo import MongoClient


def connectDB():
    client=MongoClient('mongodb://localhost:27017/')
    db=client.sponono
    return db

def export():
    f=open("/home/joo9245/spononoBackend/reported.txt",'w')
    reported=db.reported.find()
    for r in reported:
        isSpoiler="0"
        if(r["spoiler"]>r["noSpoiler"]):
            isSpoiler="1"
        f.write(r["text"]+"\t"+isSpoiler+"\n")
    f.close()

db=connectDB()
export()

