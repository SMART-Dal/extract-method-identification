import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('./train_losses_lr_768.pkl', 'rb') as f:
    train_loss = pickle.load(f)

with open('./val_losses_lr_768.pkl', 'rb') as f:
    val_loss = pickle.load(f)

window_size =5
# smooth_train = np.convolve(train_loss, np.ones(window_size) / window_size, mode='same')
# smooth_val = np.convolve(val_loss, np.ones(window_size) / window_size, mode='same')

plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Plot of Training Loss vs Validation Loss')
plt.legend()
plt.savefig('plot_lr.png')
plt.show()
