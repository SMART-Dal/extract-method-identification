import torch, numpy as np, pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from bert_based import Bert
from autoencoder_pn import Autoencoder
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        x = torch.from_numpy(x)
        y = torch.from_numpy(np.array(y)).float()   # PyTorch expects ndarray. We could have reshaped it and sent it also
        return x, y

def load_autoencoder_model(model_path, n_inputs, encoding_dim, device):
    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder

def get_bottleneck_representation(em, input_dim, encoding_dim, autoencoder):

    # model_path = "./trained_models/autoencoder_gc_pn_128_150.pth"  # Path to the saved model
    # n_inputs = input_dim  # Input dimension
    encoding_dim = encoding_dim  # Latent space dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # autoencoder = load_autoencoder_model(model_path, n_inputs, encoding_dim, device)
    with torch.no_grad():
        bottleneck_representation = autoencoder.encoder(em)
    return bottleneck_representation

def get_input_embeddings(bert, train_data, autoencoder):

    bottleneck_rep_list = []
    print("Tokenization started")
    tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in train_data]
    batch_size = 8
    num_samples = len(tokenized_data)
    bottleneck_rep_list = []

    for i in tqdm(range(0, num_samples, batch_size)):

        batch_tokenized_data = tokenized_data[i:i+batch_size] 
        input_ids = torch.tensor(batch_tokenized_data).to(device)
        # print(input_ids.shape)  
        with torch.cuda.amp.autocast():
            batch_embeddings = bert.generate_embeddings(input_ids)

        # bottleneck_rep = get_bottleneck_representation(batch_embeddings,768,128,autoencoder)

        # bottleneck_rep_list.append(bottleneck_rep.cpu())        
        bottleneck_rep_list.append(batch_embeddings.cpu())        
        # print(torch.cuda.memory_allocated()/1024**2)

    bottleneck_rep_arr = np.concatenate(bottleneck_rep_list, axis=0)

    return bottleneck_rep_arr

def model_train(input_size,hidden_size,batch_size, num_epochs, train_data, train_label, val_data, val_label, ae_path):

    learning_rate = 1e-5

    bert = Bert("microsoft/graphcodebert-base")
    trained_ae = load_autoencoder_model(ae_path,768,128, device)

    model = BinaryClassifier(input_size, hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loaders for training and validation
    print("Embedding and Latent Space extraction for train data")
    train_arr = get_input_embeddings(bert,train_data,autoencoder=trained_ae)    # Could be a part of CustomDataset
    print("Train Shape - ", train_arr.shape)
    train_dataset = CustomDataset(train_arr, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Embedding and Latent Space extraction for validation data")
    val_arr = get_input_embeddings(bert,val_data,autoencoder=trained_ae)
    valid_dataset = CustomDataset(val_arr, val_label)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print("Start training...")
    train_losses, val_losses = [], []
    for epoch in tqdm(range(num_epochs)):

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            targets = targets.unsqueeze(1) # From torch.Size([16]) to torch.Size([16,1])
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train_loss += loss.item() * inputs.size(0)
            train_loss += loss.item()
        # train_loss /= len(train_loader.dataset)

        alpha = len(train_loader) // batch_size
        epoch_train_loss = train_loss / alpha
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                targets = targets.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # valid_loss += loss.item() * inputs.size(0)
                valid_loss += loss.item()
            # valid_loss /= len(valid_loader.dataset)

            alpha = len(valid_loader) // batch_size
            epoch_val_loss = valid_loss / alpha
            val_losses.append(epoch_val_loss)

        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {round(epoch_train_loss,3)}, Valid Loss: {round(epoch_val_loss,3)}")
    
    # Save Model
    checkpoint_path = f"./trained_models/Classification/models/nn3.pth"
    torch.save(model.state_dict(), checkpoint_path)    

    with open("./trained_models/Classification/losses/train_loss_nn3.pkl","wb") as f:
        pickle.dump(train_losses,f) 
    with open("./trained_models/Classification/losses/val_loss_nn3.pkl","wb") as f:
        pickle.dump(val_losses,f)     

def model_test(test_data, test_label, nn_model_path, ae_path):

    print(test_label.shape)

    bert = Bert("microsoft/graphcodebert-base")
    trained_ae = load_autoencoder_model(ae_path,768,128, device)
    tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in test_data]
    batch_size = 8
    num_samples = len(tokenized_data)
    # bottleneck_rep_list = []
    pred_list = []

    print("Tokenization Done. Num of Samples - ", num_samples)

    nn_model = BinaryClassifier(128,50)
    nn_model.load_state_dict(torch.load(nn_model_path))
    nn_model.to(device)
    nn_model.eval()

    for i in tqdm(range(0, num_samples, batch_size)):

        batch_tokenized_data = tokenized_data[i:i+batch_size]
        input_ids = torch.tensor(batch_tokenized_data).to(device)
        with torch.cuda.amp.autocast():
            batch_embeddings = bert.generate_embeddings(input_ids)
        bottleneck_rep = get_bottleneck_representation(batch_embeddings,768,128,trained_ae)
        with torch.no_grad():
            pred_vlaues = nn_model(bottleneck_rep)
            
            pred_labels = torch.round(pred_vlaues)
            # print(pred_labels.shape)
        
        # pred_list.append(pred_labels.cpu().numpy().flatten().tolist())
        pred_list+=pred_labels.cpu().numpy().flatten().tolist()
        # print(len(pred_list))
        
        # bottleneck_rep_list.append(pred_labels.cpu())

    # print(len(pred_list))
    pred_labels_arr = np.array(pred_list)
    print(pred_labels_arr.shape)
    # y_pred = pred_labels_arr.astype(int)
    # y_pred.shape

    try:
        assert test_label.shape == pred_labels_arr.shape
    except Exception as e:
        print(test_label.shape,pred_labels_arr.shape)

    print(classification_report(test_label, pred_labels_arr))
    # print("Label shape - ", test_label.shape)
    # print("Input shape - ", bottleneck_rep_arr.shape)

    # print(pred_labels_arr[0])







if __name__=="__main__":

    with open("../data/np_arrays/train_data_file_0001.npy","+rb") as f:
        train_data_arr = np.load(f)

    with open("../data/np_arrays/train_label_file_0001.npy","+rb") as f:
        train_label_arr = np.load(f)

    with open("../data/np_arrays/test_data_file_0001.npy","+rb") as f:
        test_data_arr = np.load(f)

    with open("../data/np_arrays/test_label_file_0001.npy","+rb") as f:
        test_label_arr = np.load(f)
    print("Data loading complete")
    model_train(768,50,16,200,train_data_arr,train_label_arr,test_data_arr,test_label_arr,
                 "./trained_models/AE/models/autoencoder_gc_pn_128.pth")


    