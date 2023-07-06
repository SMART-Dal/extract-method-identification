import matplotlib.pyplot as plt
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

def loss_plot():

    with open('/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/deep-learning/trained_models/AE/losses/train_losses_128_pn.pkl', 'rb') as f:
        train_loss = pickle.load(f)

    with open('/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/deep-learning/trained_models/AE/losses/val_losses_128_pn.pkl', 'rb') as f:
        val_loss = pickle.load(f)

    # window_size =5
    # smooth_train = np.convolve(train_loss, np.ones(window_size) / window_size, mode='same')
    # smooth_val = np.convolve(val_loss, np.ones(window_size) / window_size, mode='same')

    # plt.plot(train_loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Plot of Training Loss vs Validation Loss')
    # plt.legend()
    # plt.savefig('plot_nn.png')
    # plt.show()

    # # Sample data
    # epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    epochs = list(range(1,51))
    # train_loss = [0.8, 0.6, 0.4, 0.35, 0.3, 0.25, 0.22, 0.2, 0.18, 0.16]
    # val_loss = [0.9, 0.7, 0.5, 0.45, 0.4, 0.35, 0.32, 0.3, 0.28, 0.26]

    # Find minimum validation loss and its epoch
    min_val_loss = min(val_loss)
    min_val_loss_epoch = epochs[val_loss.index(min_val_loss)]

    # Plotting the curves
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Training loss curve
    plt.plot(epochs, train_loss, marker='o', linestyle='-', color='blue', linewidth=2, label='Train Loss')

    # Validation loss curve
    plt.plot(epochs, val_loss, marker='o', linestyle='-', color='orange', linewidth=2, label='Validation Loss')

    # Title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Legend
    plt.legend(loc='best')

    # Grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for specific data points
    plt.annotate(f'Min Val Loss: {min_val_loss:.2f}', xy=(min_val_loss_epoch, min_val_loss), xytext=(min_val_loss_epoch+1, min_val_loss+0.05),
                arrowprops=dict(arrowstyle='->', color='black'), color='black')

    # Display plot
    plt.savefig('plot_loss_ae.png')
    plt.show()


def bar_plot():

    plt.rcParams['font.family'] = 'Liberation Sans'

    # Sample data
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    embeddings_ae = [0.87, 0.90, 0.87, 0.88]
    embeddings_only = [0.66, 0.84, 0.67, 0.71]

    # Set the height of the bars
    bar_height = 0.35

    # Set the position of the bars on the y-axis
    r1 = np.arange(len(categories))
    r2 = [y + bar_height for y in r1]

    # Define color palette
    # color_palette = ['#7DB3FF', '#85E3FF', '#B5FFD6', '#FFD08A']

    # Plotting the bar plot
    plt.figure(figsize=(10, 6))

    # Embeddings+AE bars
    plt.barh(r1, embeddings_ae, color='#85E3FF', height=bar_height, edgecolor='white', alpha=0.8, label='Embeddings+AE')

    # Embeddings Only bars
    plt.barh(r2, embeddings_only, color='#B5FFD6', height=bar_height, edgecolor='white', alpha=0.8, label='Embeddings Only')

    # Customize the plot
    plt.title('Metrics Comparison: Embeddings+AE vs Embeddings Only', fontweight='bold', fontsize=14)
    plt.xlabel('Score', fontweight='bold', fontsize=12)
    plt.ylabel('Metrics', fontweight='bold', fontsize=12)
    plt.yticks([r + bar_height/2 for r in range(len(categories))], categories, fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlim(0, 1)

    # Add data labels to the bars
    for i, v in enumerate(embeddings_ae):
        plt.text(v + 0.03, i - 0.02, str(v), color='black', fontweight='bold', fontsize=10)
    for i, v in enumerate(embeddings_only):
        plt.text(v + 0.03, i + bar_height - 0.02, str(v), color='black', fontweight='bold', fontsize=10)

    # Remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust legend position inside the x-y chart
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gca().transAxes)

    # Set the background color
    plt.gca().set_facecolor('#F5F5F5')
    plt.gca().set_xlim(right=1.2)

    plt.savefig('plot_metric_compare.png')


def bar_plot_mod():
    plt.rcParams['font.family'] = 'Liberation Sans'

    # Sample data
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    embeddings_ae = [0.87, 0.90, 0.87, 0.88]
    embeddings_only = [0.66, 0.84, 0.67, 0.71]

    # Set the height of the bars
    bar_height = 0.35

    # Set the position of the bars on the y-axis
    r1 = np.arange(len(categories))
    r2 = [y + bar_height for y in r1]

    # Define color palette
    grayscale_cmap = cm.get_cmap('gray')

    # Plotting the bar plot
    plt.figure(figsize=(10, 6))

    # Embeddings+AE bars
    plt.barh(r1, embeddings_ae, color=grayscale_cmap(0.2), height=bar_height, edgecolor='white', alpha=0.8, label='Embeddings+AE')

    # Embeddings Only bars
    plt.barh(r2, embeddings_only, color=grayscale_cmap(0.6), height=bar_height, edgecolor='white', alpha=0.8, label='Embeddings Only')

    # Customize the plot
    # plt.title('Metrics Comparison: Embeddings+AE vs Embeddings Only', fontweight='bold', fontsize=14)
    plt.xlabel('Score', fontweight='bold', fontsize=16)
    plt.ylabel('Metrics', fontweight='bold', fontsize=16)
    plt.yticks([r + bar_height/2 for r in range(len(categories))], categories, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim(0, 1)

    # Add data labels to the bars
    for i, v in enumerate(embeddings_ae):
        plt.text(v + 0.03, i - 0.02, str(v), color='black', fontweight='bold', fontsize=16)
    for i, v in enumerate(embeddings_only):
        plt.text(v + 0.03, i + bar_height - 0.02, str(v), color='black', fontweight='bold', fontsize=16)

    # Remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust legend position inside the x-y chart
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=plt.gca().transAxes, fontsize=10)

    # Set the background color to transparent
    plt.gca().set_facecolor('none')
    plt.gca().set_xlim(right=1.2)

    plt.savefig('plot_metric_compare.png', transparent=True)



if __name__=="__main__":
    # bar_plot_mod()
    loss_plot()