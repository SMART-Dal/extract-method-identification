import torch, json, pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from transformers import BertModel, BertTokenizer
from bert_based import Bert

class Autoencoder(nn.Module):
    def __init__(self, n_inputs, encoding_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_inputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def __get_data_from_jsonl(data_file):

    data = []
    max_p = 0
    with open(data_file, 'r') as file:
        for line in file:
            item = json.loads(line)
            if len(item['positive_case_methods'])==0:
                continue 
            if len(item['positive_case_methods'])>max_p:
                max_p=len(item['positive_case_methods'])
            data+=item['positive_case_methods']

    return data

def train_autoencoder(data, batch_size, num_epochs, device="cuda", save_interval=1):
    num_batches = len(data) // batch_size

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    bert = Bert("microsoft/graphcodebert-base")

    # n_inputs = data.size(1)  # Input dimension
    n_inputs = 768
    encoding_dim = 32  # Latent space dimension

    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder = autoencoder.to(device)  # Move model to GPU
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision training scaler
    losses = []

    for epoch in range(num_epochs):
        for i, batch_data in enumerate(data_loader):
            epoch_loss = 0.0
            print(len(batch_data))
            # print(batch_data)
            # batch_data = batch_data.to(device)  # Move batch data to GPU
            tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=177) for text in batch_data]
            input_ids = torch.tensor(tokenized_data).to(device)
            # Generate embeddings using BERT
            with torch.cuda.amp.autocast():
                embeddings = bert.generate_embeddings(input_ids)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = autoencoder(embeddings)

            # Compute loss
            loss = criterion(outputs, embeddings)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, num_epochs, i + 1, num_batches, loss.item()))
            
        epoch_loss /= num_batches
        losses.append(epoch_loss)

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"autoencoder_epoch_{epoch + 1}.pth"
            torch.save(autoencoder.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")
    
    # Save losses list
    with open('losses.pkl', 'wb') as f:
        pickle.dump(losses, f)
    print(f"Training losses saved at losses.pkl")

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
    tokenized_text = bert.tokenizer.encode(code, padding='max_length', truncation=True, max_length=177)
    input_ids = torch.tensor(tokenized_text).unsqueeze(0).to(device)
    # inputs = input_ids.to(device)
    # gcb_model=bert.model.to(device)
    em=bert.generate_embeddings(input_ids)
    with torch.no_grad():
        encodings = autoencoder.encoder(em) #Start from here
        bottleneck_representation = encodings[-1]  # Get the output of the last layer (bottleneck)
    return bottleneck_representation

if __name__=="__main__":
    data = __get_data_from_jsonl("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/data/archive/file_0000.jsonl")
    train_autoencoder(data,8,10)