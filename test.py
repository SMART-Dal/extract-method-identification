import pickle
import matplotlib.pyplot as plt

with open('deep-learning/train_losses.pkl','rb') as f:
    train_losses = pickle.load(f)

with open('deep-learning/val_losses.pkl','rb') as f:
    val_losses = pickle.load(f)

plt.plot(train_losses, label='List 1')
plt.plot(val_losses, label='List 2')

plt.show() #Need to deserialize on CUDA 