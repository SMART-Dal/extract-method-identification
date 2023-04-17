from dotenv import dotenv_values
from pymongo import MongoClient
from bson.objectid import ObjectId
import subprocess,os

class Database:

    def __init__(self,collection_name) -> None:
        env_values = dotenv_values(".env")
        self.url = env_values['MONGO_CLIENT']
        self.db_name = env_values['DB_NAME']
        self.collection_name = collection_name
        self.__connect_db()

    def __connect_db(self):
        client = MongoClient(self.url)
        self.db = client[self.db_name]

    def __fetch_collection(self, collection_name: str):
        collection = self.db.get_collection(collection_name)
        return collection

    def insert_docs(self,doc_list):
        collection = self.__fetch_collection(self.collection_name)
        collection.insert_many(doc_list)

    def insert_doc(self,doc):
        collection = self.__fetch_collection(self.collection_name)
        collection.insert_one(doc)

    def find_docs(self, query,projection={}):
        collection = self.__fetch_collection(self.collection_name)
        return collection.find(query,projection)

    def estimated_doc_count(self):
        collection = self.__fetch_collection(self.collection_name)
        return collection.estimated_document_count()
        
    def update_by_id(self, doc_id, col_name: str, col_val):
        collection = self.__fetch_collection(self.collection_name)
        collection.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {col_name: col_val}}
        )
    
    def update_by_field(self, match, replacement):
        collection = self.__fetch_collection(self.collection_name)
        # collection.update_one(match,{"$set":replacement})
        collection.update_many(match,{"$set":replacement})

if __name__=="__main__":
    db = Database("test_cc")
    db.insert_docs([{"Test from CC":True}])
    os.chdir("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/executable/RefactoringMiner/bin")
    sproc = subprocess.run(["sh","RefactoringMiner","-h"],capture_output=True)
    print(sproc.stderr)