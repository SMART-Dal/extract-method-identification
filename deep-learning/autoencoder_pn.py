import torch, json, pickle, time, numpy as np, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from bert_based import Bert
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"


class Autoencoder(nn.Module):
    def __init__(self, n_inputs, encoding_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_inputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        return x

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

def get_train_val_split(data, labels):
    
    train_data, test_data, _, _ = train_test_split(data,labels, test_size=0.2, stratify=labels)
    print(f"Training sample length - {len(train_data)}. Validation Sample Length - {len(test_data)}")
    return train_data, test_data

def get_embedded_data(data):
    bert = Bert("microsoft/graphcodebert-base")
    print(len(data))
    tokenized_data = [bert.tokenizer.encode(x, padding='max_length', truncation=True, max_length=512) for x in data]
    batch_size = 2
    num_samples = len(tokenized_data)
    embedding_list = []
    print("Generating Batch Embeddings")
    for i in tqdm(range(0, num_samples, batch_size)):
        
        batch_tokenized_data = tokenized_data[i:i+batch_size]         
        input_ids = torch.tensor(batch_tokenized_data).to(device)      
        with torch.cuda.amp.autocast():
            batch_embeddings = bert.generate_embeddings(input_ids)
        embedding_list.append(batch_embeddings.cpu())

    batch_embedding_arr = torch.concatenate(embedding_list, axis=0)    
    return batch_embedding_arr


def train_autoencoder(data, batch_size, num_epochs,n_inputs,encoding_dim, save_interval=1, valid_data=None,
                       model_name="microsoft/graphcodebert-base",model_shorthand="gc"):
    num_batches = len(data) // batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = SummaryWriter("./logs/")

    print("Get the data embeddings...")
    embedded_data = get_embedded_data(data)

    train_dataset = CustomDataset(embedded_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if valid_data is not None:
        valid_dataset = CustomDataset(embedded_data)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder = autoencoder.to(device)  # Move model to GPU
    criterion = nn.MSELoss()

    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=1e-8)

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training scaler
    train_losses, val_losses = [], []

    print("Start the training...")
    for epoch in tqdm(range(num_epochs)):

        start_time = time.time()
        train_loss = 0.0
        autoencoder.train()
        for _, batch_data in enumerate(train_data_loader):

            train_input = torch.tensor(batch_data).to(device)
            optimizer.zero_grad()
            outputs = autoencoder(train_input)
            loss = criterion(outputs, train_input)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        alpha = len(train_data_loader) // batch_size
        epoch_train_loss = train_loss / alpha
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        train_losses.append(epoch_train_loss)

        if valid_data is not None:
            autoencoder.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for _, val_batch_data in enumerate(valid_data_loader):
                    val_input = torch.tensor(val_batch_data).to(device)
                    val_outputs = autoencoder(val_input)
                    val_loss = criterion(val_outputs,val_input)                    

                    val_loss_sum+=val_loss.item()

            alpha = len(valid_data_loader) // batch_size
            epoch_val_loss = val_loss_sum / alpha
            writer.add_scalar("Loss/train", epoch_val_loss, epoch)
            val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_val_loss}')
        
        print(f"Time for epoch {epoch+1} - ", time.time()-start_time)
    

    # Save Model
    checkpoint_path = f"./trained_models/AE/models/autoencoder_{model_shorthand}_pn_{encoding_dim}.pth"
    torch.save(autoencoder.state_dict(), checkpoint_path)

    # Save losses list
    with open(f'./trained_models/AE/losses/train_losses_{encoding_dim}_pn.pkl', 'wb') as f: # TODO: Name dynamic
        pickle.dump(train_losses, f)

    with open(f'./trained_models/AE/losses/val_losses_{encoding_dim}_pn.pkl', 'wb') as f:
        pickle.dump(val_losses, f)
    # print(f"Training losses saved at losses.pkl")

    return autoencoder


if __name__=="__main__":
    with open("../data/np_arrays/train_data_file_0000.npy","rb") as f:
        train_data_arr = np.load(f)

    with open("../data/np_arrays/test_data_file_0000.npy","rb") as f:
        val_data_arr = np.load(f)

    print(train_data_arr.shape)
    # train_autoencoder(train_data_arr,8,150,768,128,save_interval=50, valid_data=val_data_arr,
    #                   model_name="microsoft/codebert-base",model_shorthand="cb")
    train_autoencoder(train_data_arr,8,150,768,128,save_interval=50, valid_data=val_data_arr,
                    model_name="microsoft/graphcodebert-base",model_shorthand="gc")
    # train_autoencoder(train_data_arr,8,150,768,128,save_interval=50, valid_data=val_data_arr,
    #             model_name="Salesforce/codet5-small",model_shorthand="ct")