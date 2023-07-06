import torch, json, pickle, time, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from bert_based import Bert
from autoencoder_pn import Autoencoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from decimal import Decimal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegression, self).__init__()
        # self.dropout = nn.Dropout(.2)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # out = self.dropout(x)
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

            if len(data)>=10000:
                break            
            
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
    
    train_data, test_data, train_label, test_label = train_test_split(data,labels, test_size=0.2, stratify=labels)
    print(f"Training sample length - {len(train_data)}. Validation Sample length - {len(test_data)}")
    print(f"Training label length - {len(train_label)}. Validation label length - {len(test_label)}")
    return train_data, test_data, train_label, test_label


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


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def load_autoencoder_model(model_path, n_inputs, encoding_dim, device):
    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder

def get_bottleneck_representation(em, input_dim, encoding_dim):

    model_path = "./trained_models/autoencoder_gc_pn_128_150.pth"  # Path to the saved model
    n_inputs = input_dim  # Input dimension
    encoding_dim = encoding_dim  # Latent space dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_autoencoder_model(model_path, n_inputs, encoding_dim, device)
    with torch.no_grad():
        # encodings = autoencoder.encoder(em) #Start from here
        # print(encodings.shape)

        bottleneck_representation = autoencoder.encoder(em)
        # bottleneck_representation = encodings[-1]  # Get the output of the last layer (bottleneck)
    # print(bottleneck_representation.shape)
    return bottleneck_representation

def perform_pca(data, n_components):

    data_np = data.numpy()
    pca = PCA(n_components=n_components)
    pca.fit(data_np)
    projected_data_np = pca.transform(data_np)
    projected_data = torch.from_numpy(projected_data_np)
    return projected_data

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

            print(embeddings.shape)
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

def train_lr_ae(data, labels, input_dim,num_classes, val_data=None, val_labels=None):


    learning_rate = 1e-5
    batch_size = 32
    num_epochs = 150

    bert = Bert("microsoft/graphcodebert-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # data = torch.randn(1000, input_dim)
    # labels = torch.randint(0, num_classes, (1000,))

    # dataset = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(labels))
    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if val_data:
        val_dataset = CustomDataset(val_data,val_labels)
        val_dataloader = DataLoader(val_dataset,batch_size=batch_size, shuffle=True)

    model = LogisticRegression(input_dim, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    scaler = torch.cuda.amp.GradScaler()
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):

        start_time = time.time()

        model.train()
        train_loss=0.0
        for inputs, targets in dataloader:            
            # batch_data = batch_data.to(device)  # Move batch data to GPU
            tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in inputs]
            input_ids = torch.tensor(tokenized_data).to(device)

            with torch.cuda.amp.autocast():
                embeddings = bert.generate_embeddings(input_ids)

            embeddings = embeddings.to(device)

            bottleneck_rep = get_bottleneck_representation(embeddings,768,128)

            targets = targets.to(device)
            
            optimizer.zero_grad()
            # outputs = model(embeddings)
            with torch.cuda.amp.autocast():
                outputs = model(bottleneck_rep)
                loss = criterion(outputs, targets)

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss+=loss.item()
        
        # train_losses.append(loss.item())

        if val_data:
            model.eval()
            val_loss_sum=0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_dataloader:
                    val_tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in val_inputs]
                    val_input_ids = torch.tensor(val_tokenized_data).to(device)
                    torch.cuda.empty_cache()
                    with torch.cuda.amp.autocast():
                        val_embeddings = bert.generate_embeddings(val_input_ids)

                    val_embeddings = val_embeddings.to(device)
                    val_bottleneck_rep = get_bottleneck_representation(val_embeddings,768, 128)
                    val_targets = val_targets.to(device)
                    
                    val_outputs = model(val_bottleneck_rep)
                    val_loss = criterion(val_outputs,val_targets)

                    val_loss_sum+=val_loss.item()
            # val_losses.append(val_loss.item())    
        
        
            alpha = len(val_dataloader) // batch_size
            epoch_val_loss = val_loss_sum / alpha
            val_losses.append(epoch_val_loss)                 

        alpha = len(dataloader) // batch_size
        epoch_train_loss = train_loss / alpha
        train_losses.append(epoch_train_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_val_loss}')
        print(f"Time for epoch {epoch+1} - ", time.time()-start_time)
        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()} \t\t Validation Loss: {val_loss.item()}')

    # Save losses list
    with open(f'./metrics/train_losses_lr_ae_{input_dim}.pkl', 'wb') as f:
        pickle.dump(train_losses, f)

    with open(f'./metrics/val_losses_lr_ae_{input_dim}.pkl', 'wb') as f:
        pickle.dump(val_losses, f)
    # Save the trained model
    torch.save(model.state_dict(), f'./trained_models/logistic_regression_model_ae_{input_dim}.pth')

def train_lr_pca(data, labels, input_dim,num_classes):


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

            embeddings_pca = perform_pca(embeddings,2)

            targets = targets.to(device)

            outputs = model(embeddings_pca)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), './trained_models/logistic_regression_model.pth')


if __name__=="__main__":
    # TODO - Change get_bottleneck
    data, labels = __get_data_from_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/data/output/file_0001.jsonl")
    train_data, test_data, train_label, test_label = get_train_test_val_split(data,labels)
    # train_lr(data, labels, 768, 2)
    train_lr_ae(train_data, train_label, 128, 2, test_data, test_label)
    # train_lr_pca(data, labels, 2, 2)