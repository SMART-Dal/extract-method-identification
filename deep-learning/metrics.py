import torch, json
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from classify import LogisticRegression, get_bottleneck_representation
from bert_based import Bert

def __get_data_from_jsonl(data_file):

    data, labels = [], []
    max_p, max_n = 0, 0
    with open(data_file, 'r') as file:
        for line in file:
            item = json.loads(line)
            if len(item['positive_case_methods'])==0:
                continue 
            if len(item['positive_case_methods'])>max_p:
                max_p=len(item['positive_case_methods'])
            data+=item['positive_case_methods']
            # labels+=[1 for i in range(len(item))]
            labels.extend([1]*len(item['positive_case_methods']))
            data+=item['negative_case_methods']
            # labels+=[1 for i in range(len(item))]
            labels.extend([0]*len(item['positive_case_methods']))
        try:
            assert len(labels)==len(data)
        except AssertionError as e:
            print(len(labels))
            print(len(data))
    print("Total samples - ", len(data))
    print("Maximum methods per case in a repo - ", max_p)
    return data, labels


def get_train_test_val_split(data, labels):
    
    train_data, test_data, train_label, test_label = train_test_split(data,labels, test_size=0.2, stratify=labels, random_state=42)
    print(f"Training sample length - {len(train_data)}. Validation Sample length - {len(test_data)}")
    print(f"Training label length - {len(train_label)}. Validation label length - {len(test_label)}")
    return train_data, test_data, train_label, test_label


def test_lr(data, labels, input_dim, num_classes, model_path, bool_ae):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LogisticRegression(input_dim,num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    bert = Bert("microsoft/graphcodebert-base")
    # Set the model in evaluation mode
    model.eval()

    predicted_labels = []
    true_labels = []
    batch_size = 16

    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        batch_texts = data[i:i+batch_size]

        val_tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in batch_texts]
        val_input_ids = torch.tensor(val_tokenized_data).to(device)

        torch.cuda.empty_cache() # Won't work with CPU
        with torch.cuda.amp.autocast():
            val_embeddings = bert.generate_embeddings(val_input_ids)    

        val_embeddings = val_embeddings.to(device)
        
        if bool_ae:
            input = get_bottleneck_representation(val_embeddings, 768,input_dim )
        else:
            input = val_embeddings

        labels = torch.tensor(labels).to(device)

        with torch.no_grad():
            outputs = model(input)
            batch_predicted_labels = torch.argmax(outputs, dim=1).cpu()

        predicted_labels.extend(batch_predicted_labels)
        true_labels.extend(labels[i:i+batch_size].cpu())
    
    predicted_labels = torch.stack(predicted_labels)
    true_labels = torch.stack(true_labels)
    
    accuracy = (predicted_labels == true_labels).sum().item() / true_labels.size(0)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    return accuracy, precision, recall, f1

if __name__=="__main__":

    data, labels = __get_data_from_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/data/output/file_0001.jsonl")
    train_data, test_data, train_label, test_label = get_train_test_val_split(data,labels)
    accuracy, precision, recall, f1 = test_lr(test_data,test_label,128,2,
                                         "./trained_models/logistic_regression_model_ae_128.pth",
                                        #  "./trained_models/logistic_regression_model_768.pth",
                                         True
                                         )
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)
    
