import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('./train_losses.pkl', 'rb') as f:
    train_loss = pickle.load(f)

with open('./val_losses.pkl', 'rb') as f:
    val_loss = pickle.load(f)

window_size =5
# smooth_train = np.convolve(train_loss, np.ones(window_size) / window_size, mode='same')
# smooth_val = np.convolve(val_loss, np.ones(window_size) / window_size, mode='same')

plt.plot(train_loss, label='List 1')
plt.plot(val_loss, label='List 2')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of List 1 and List 2')
plt.legend()
plt.savefig('plot_ae.png')
plt.show()
