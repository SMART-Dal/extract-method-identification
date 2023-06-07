from transformers import  AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

class Bert:

    def __init__(self, model_name) -> None:
        # model_name = "microsoft/graphcodebert-base"
        if model_name == "microsoft/graphcodebert-base":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModel.from_pretrained(model_name)
        elif model_name == "microsoft/codebert-base":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Irrelevant
    def generate_embeddings(self):
        database = Database("refactoring_details_neg")
        # database.connect_db()
        # collection = database.fetch_collection("refactoring_information")
        # collection_len = collection.estimated_document_count()
        collection_len = database.estimated_doc_count()


        doc_count = 1
        for doc in database.find_docs({}, {"_id": 1, "method_refactored": 1, "meth_rf_neg":1}):
            doc_id = doc["_id"]
            code_snippet = doc["method_refactored"]
            code_snippet_neg = doc["meth_rf_neg"]
            print(f'Generating embedding for doc_id:{doc_id} | Count-{doc_count}...')
            
            # Compute embeddings
            tokenized_input_pos = self.tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True)
            output = self.model(**tokenized_input_pos)
            embedding_pos = output.last_hidden_state.mean(dim=1).squeeze().tolist()

            #Neg Embedding
            tokenized_input_neg = self.tokenizer(code_snippet_neg, return_tensors="pt", padding=True, truncation=True)
            output = self.model(**tokenized_input_neg)
            embedding_neg = output.last_hidden_state.mean(dim=1).squeeze().tolist()

            # Update document in MongoDB with embedding
            database.update_by_id(doc_id, "embedding_pos", embedding_pos)
            database.update_by_id(doc_id,"embedding_neg", embedding_neg)

            collection_len -= 1
            doc_count += 1
            print(f'Embedding added for doc_id:{doc_id} | Remaining: {collection_len}.')

    
    def generate_individual_embedding(self,code_snippet):
        tokenized_input_pos = self.tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True)
        output = self.model(**tokenized_input_pos)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()

        return embedding
    
    
    def embedding_generator(self,data_batch):
        for item in data_batch:
            positive_case_methods = item['positive_case_methods']
            negative_case_methods = item['negative_case_methods']

            positive_embeddings = self.generate_individual_embedding(positive_case_methods)
            negative_embeddings = self.generate_individual_embedding(negative_case_methods)

            yield positive_embeddings, negative_embeddings


    def gen_embeddings(self,code):

        tokenized_input_pos = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = self.model(**tokenized_input_pos)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().tolist()
        if len(code)==1:
            return [embedding]
        else:
            return embedding




if __name__=="__main__":
    Bert().generate_embeddings()