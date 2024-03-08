import torch, json, pickle, time, numpy as np, sys, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from bert_based import Bert
from tqdm import tqdm

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


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


def get_embedded_data(data):
    bert = Bert("microsoft/graphcodebert-base")
    print(len(data))
    tokenized_data = [bert.tokenizer.encode(x, padding='max_length', truncation=True, max_length=512) for x in data]
    batch_size = 8
    num_samples = len(tokenized_data)
    embedding_list = []
    print("Generating Batch Embeddings")
    for i in tqdm(range(0, num_samples, batch_size)):
        batch_tokenized_data = tokenized_data[i:i + batch_size]
        input_ids = torch.tensor(batch_tokenized_data).to(device)
        with torch.cuda.amp.autocast():
            batch_embeddings = bert.generate_embeddings(input_ids, device=device)
        expected_shape = (batch_size, 768)
        if expected_shape == batch_embeddings.cpu().shape:
            embedding_list.append(batch_embeddings.cpu())
        else:
            embedding_list.append(torch.reshape(batch_embeddings.cpu(), (-1,768)))
        # print("Embedding list shape", batch_embeddings.cpu().shape)
    batch_embedding_arr = torch.cat(embedding_list, dim=0)
    return batch_embedding_arr



def train_autoencoder(data, batch_size, num_epochs, n_inputs, encoding_dim,
                      out_folder_path, save_interval=1, val_data=None,
                      model_name="microsoft/graphcodebert-base", model_shorthand="gc"):

    print("Get the data embeddings...")
    embedded_data = get_embedded_data(data)

    train_dataset = CustomDataset(embedded_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data is not None:
        embedded_val_data = get_embedded_data(val_data)
        valid_dataset = CustomDataset(embedded_val_data)
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
        train_losses.append(epoch_train_loss)

        if val_data is not None:
            autoencoder.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for _, val_batch_data in enumerate(valid_data_loader):
                    val_input = torch.tensor(val_batch_data).to(device)
                    val_outputs = autoencoder(val_input)
                    val_loss = criterion(val_outputs, val_input)

                    val_loss_sum += val_loss.item()

            alpha = len(valid_data_loader) // batch_size
            epoch_val_loss = val_loss_sum / alpha
            val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch + 1} \t\t Training Loss: {epoch_train_loss} \t\t Validation Loss: {epoch_val_loss}')
        print(f"Time for epoch {epoch + 1} - ", time.time() - start_time)

    # Save Model
    checkpoint_folder_path = os.path.join(out_folder_path,
                                          "trained_models", "AE", "models")
    os.makedirs(checkpoint_folder_path, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder_path, f"autoencoder_{model_shorthand}_pn_{encoding_dim}.pth")
    torch.save(autoencoder.state_dict(), checkpoint_path)

    # Save losses list
    losses_folder_path = os.path.join(out_folder_path,
                                      "trained_models", "AE", "losses")
    os.makedirs(losses_folder_path, exist_ok=True)
    with open(os.path.join(losses_folder_path, f'train_losses_{encoding_dim}_pn.pkl'), 'wb') as f:
        pickle.dump(train_losses, f)

    with open(os.path.join(losses_folder_path, f'val_losses_{encoding_dim}_pn.pkl'), 'wb') as f:
        pickle.dump(val_losses, f)
    print(f"Training losses saved.")
    return autoencoder


if __name__ == "__main__":
    print("Start")

    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    folder_path = os.path.dirname(train_file_path)
    with open(train_file_path, "rb") as f:
        train_data_arr = np.load(f)

    with open(test_file_path, "rb") as f:
        val_data_arr = np.load(f)

    # train_data_arr = train_data_arr[int(len(train_data_arr)/8)]
    # val_data_arr = val_data_arr[int(len(val_data_arr)/8)]
    print("Train Data Shape", train_data_arr.shape)
    print("Validation Data Shape", val_data_arr.shape)
    train_autoencoder(train_data_arr, 8, 50, 768, 128,
                      out_folder_path=folder_path,
                      save_interval=50, val_data=val_data_arr,
                      model_name="microsoft/graphcodebert-base", model_shorthand="gc")
