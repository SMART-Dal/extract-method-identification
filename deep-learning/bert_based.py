from transformers import  AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch, numpy as np

class Bert:

    def __init__(self, model_name) -> None:
        # model_name = "microsoft/graphcodebert-base"
        self.model_name=model_name
        if model_name == "microsoft/graphcodebert-base":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model=AutoModel.from_pretrained(model_name)
        elif model_name == "microsoft/codebert-base":
            self.tokenizer= AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def generate_embeddings(self,code,device="cuda"):

        inputs = code.to(device)
        model = self.model.to(device)
        outputs = model(inputs)
        # print(type(outputs))
        # print(outputs.__dict__)
        if self.model_name=="microsoft/graphcodebert-base":
            with torch.no_grad():
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return embeddings



if __name__=="__main__":
    Bert().generate_embeddings()