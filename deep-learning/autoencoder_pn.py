import torch, json, pickle, time, numpy as np, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from bert_based import Bert
from sklearn.model_selection import train_test_split

# class Autoencoder(nn.Module):
#     def __init__(self, n_inputs, encoding_dim):
#         super(Autoencoder, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(n_inputs, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, encoding_dim),
#             nn.ReLU()
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, n_inputs),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
    
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



def train_autoencoder(data, batch_size, num_epochs,n_inputs,encoding_dim, save_interval=1, valid_data=None, model_shorthand="gc"):
    num_batches = len(data) // batch_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    if valid_data:
        valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)

    bert = Bert("microsoft/graphcodebert-base")
    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder = autoencoder.to(device)  # Move model to GPU
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-4, weight_decay=1e-8)

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training scaler
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):

        start_time = time.time()
        train_loss = 0.0
        autoencoder.train()
        for _, batch_data in enumerate(train_data_loader):

            # batch_data = batch_data.to(device)  # Move batch data to GPU
            if _ == 0:
                print(type(batch_data)) 

            # print(batch_data[0])
            tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in batch_data]
            input_ids = torch.tensor(tokenized_data).to(device)

            with torch.cuda.amp.autocast():
                embeddings = bert.generate_embeddings(input_ids)

            optimizer.zero_grad()
            outputs = autoencoder(embeddings)
            loss = criterion(outputs, embeddings)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # epoch_train_loss = train_loss / len(train_data_loader)
        alpha = len(train_data_loader) // batch_size
        epoch_train_loss = train_loss / alpha
        train_losses.append(epoch_train_loss)

        if valid_data:
            autoencoder.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for _, val_batch_data in enumerate(valid_data_loader):
                    val_tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in val_batch_data]
                    val_input_ids = torch.tensor(val_tokenized_data).to(device)
                    torch.cuda.empty_cache()
                    with torch.cuda.amp.autocast():
                        val_embeddings = bert.generate_embeddings(val_input_ids)

                    val_outputs = autoencoder(val_embeddings)
                    val_loss = criterion(val_outputs,val_embeddings)

                    val_loss_sum+=val_loss.item()
            
            # epoch_val_loss = val_loss / len(valid_data_loader)
            alpha = len(valid_data_loader) // batch_size
            epoch_val_loss = val_loss_sum / alpha
            val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1} \t\t Training Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_val_loss}')

        # print('Epoch [{}/{}], Loss: {:.4f}'.format(
        #     epoch + 1, num_epochs, train_loss/len(train_data_loader)))
        
        print(f"Time for epoch {epoch+1} - ", time.time()-start_time)

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"./trained_models/autoencoder_{model_shorthand}_pn_{encoding_dim}_{epoch + 1}.pth"
            torch.save(autoencoder.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")
    
    # Save losses list
    with open(f'train_losses_{encoding_dim}_pn.pkl', 'wb') as f: # TODO: Name dynamic
        pickle.dump(train_losses, f)

    with open(f'val_losses_{encoding_dim}_pn.pkl', 'wb') as f:
        pickle.dump(val_losses, f)
    # print(f"Training losses saved at losses.pkl")

    return autoencoder


def load_autoencoder_model(model_path, n_inputs, encoding_dim, device):
    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder

def get_bottleneck_representation(code,device="cuda"):

    model_path = "autoencoder_epoch_1.pth"  # Path to the saved model
    n_inputs = 768  # Input dimension
    encoding_dim = 32  # Latent space dimension
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_autoencoder_model(model_path, n_inputs, encoding_dim, device)


    bert = Bert("microsoft/graphcodebert-base")
    tokenized_text = bert.tokenizer.encode(code, padding='max_length', truncation=True, max_length=177) # Works for single code
    input_ids = torch.tensor(tokenized_text).unsqueeze(0).to(device)
    # inputs = input_ids.to(device)
    # gcb_model=bert.model.to(device)
    em=bert.generate_embeddings(input_ids)
    with torch.no_grad():
        encodings = autoencoder.encoder(em) #Start from here
        bottleneck_representation = encodings[-1]  # Get the output of the last layer (bottleneck)
    return bottleneck_representation

if __name__=="__main__":
    data, labels = __get_data_from_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/data/archive/file_0000.jsonl")
    train_data, valid_data = get_train_val_split(data, labels)
    train_autoencoder(train_data,8,150,768,128,save_interval=10, valid_data=valid_data,model_shorthand="gc") # TODO Learn with 1 epoch