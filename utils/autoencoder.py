import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, bottleneck_size):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_size),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid(),  # Use Sigmoid activation for reconstruction in the range [0, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Set hyperparameters
input_size = 768  # Assuming the text input is represented as a vector of length 768
bottleneck_size = 32
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load your dataframe and split it into training and validation sets
# Assuming your dataframe is called 'df' and the text input column is 'text' and label column is 'label'
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Preprocess the text input column using CountVectorizer or any other suitable method
vectorizer = CountVectorizer(max_features=input_size)
train_text = vectorizer.fit_transform(train_df['text']).toarray()
val_text = vectorizer.transform(val_df['text']).toarray()

# Create DataLoader for training and validation
train_dataset = TensorDataset(torch.tensor(train_text, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(val_text, dtype=torch.float32))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Create the combined model with autoencoder and classifier
class CombinedModel(nn.Module):
    def __init__(self, autoencoder, classifier):
        super(CombinedModel, self).__init__()
        self.autoencoder = autoencoder
        self.classifier = classifier
    
    def forward(self, x):
        encoded = self.autoencoder.encoder(x)
        output = self.classifier(encoded)
        return output

# Create the autoencoder model
autoencoder = Autoencoder(input_size, bottleneck_size)

# Create the classifier model
num_classes = 2  # Assuming 2 classes: positive and negative
classifier = nn.Sequential(
    nn.Linear(bottleneck_size, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
)

# Create the combined model
model = CombinedModel(autoencoder, classifier)

# Define the loss functions and optimizer
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    
    # Training
    model.train()
    for batch in train_dataloader:
        inputs = batch[0]
        targets = train_df['label']
        
        # Forward pass
        outputs = model(inputs)
        
        # Reconstruction loss
        reconstruction_loss = reconstruction_criterion(outputs, inputs)
        
        # Classification loss
        classification_loss = classification_criterion(outputs, targets)
        
        # Combined loss
        loss = reconstruction_loss + classification_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch[0]
            targets = val_df['label']
            
            # Forward pass
            outputs = model(inputs)
            
            # Reconstruction loss
            reconstruction_loss = reconstruction_criterion(outputs, inputs)
            
            # Classification loss
            classification_loss = classification_criterion(outputs, targets)
            
            # Combined loss
            loss = reconstruction_loss + classification_loss
            
            val_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}")

# Now your combined model is trained and can be used for inference
