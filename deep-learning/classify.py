import torch, json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from bert_based import Bert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

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

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


def train_lr(data, labels, input_dim,num_classes):


    learning_rate = 0.001
    batch_size = 16
    num_epochs = 10

    bert = Bert("microsoft/graphcodebert-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data = torch.randn(1000, input_dim)
    # labels = torch.randint(0, num_classes, (1000,))

    # dataset = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels))
    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LogisticRegression(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:            
            # batch_data = batch_data.to(device)  # Move batch data to GPU
            tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in inputs]
            input_ids = torch.tensor(tokenized_data).to(device)

            with torch.cuda.amp.autocast():
                embeddings = bert.generate_embeddings(input_ids)

            embeddings = embeddings.to(device)
            targets = targets.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), './trained_models/logistic_regression_model.pth')


if __name__=="__main__":
    data, labels = __get_data_from_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/data/output/file_0001.jsonl")
    train_lr(data, labels, 768, 2)