import matplotlib.pyplot as plt
import pickle, sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn import manifold

def loss_plot(train_loss_path, val_loss_path, epochs):

    with open(train_loss_path, 'rb') as f:
        train_loss = pickle.load(f)

    with open(val_loss_path, 'rb') as f:
        val_loss = pickle.load(f)

    epochs = list(range(1,epochs+1))

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


'''
The following 3 methods have been borrowed from this repository - https://github.com/VulDetProject/ReVeal/tree/ca31b783384b4cdb09b69950e48f79fa0748ef1d
'''
def plot_embedding_tsne(X_org, y, title=None):
    X, Y = X_org, y
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    print('Fitting TSNE!')
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    file_ = open(str(title) + '-tsne-features.json', 'w')
    if isinstance(X, np.ndarray):
        _x = X.tolist()
        _y = Y.tolist()
    else:
        _x = X
        _y = Y
    file_.close()
    plt.figure(title)
    c0, c1 = 0,0
    tsne_feat = np.zeros(X.shape)

    for i in range(X.shape[0]):
        if Y[i] == 0:
          plt.text(X[i, 0], X[i, 1], 'o',
                     fontdict={'weight': 'bold', 'size': 9})
        else:
          plt.text(X[i, 0], X[i, 1], '+',
                     color=plt.cm.Set1(0),
                     fontdict={'weight': 'bold', 'size': 9})
          
    os.makedirs('./logs/tsne/', exist_ok=True)
    np.save('tsne_feat.npy',tsne_feat)
    print(c0,c1)
    if title is not None:
        plt.title("")
    plt.show()

def calculate_centroids(_features, _labels):
    pos = []
    neg = []
    for f, l  in zip(_features, _labels):
        if l == 1:
            pos.append(f)
        else:
            neg.append(f)
    posx = [x[0] for x in pos]
    posy = [x[1] for x in pos]
    negx = [x[0] for x in neg]
    negy = [x[1] for x in neg]
    _px = np.median(posx)
    _py = np.median(posy)
    _nx = np.median(negx)
    _ny = np.median(negy)
    return (_px, _py), (_nx, _ny)

def calculate_distance(p1, p2):
    return np.abs(np.sqrt(((p1[0] - p2[0])*(p1[0] - p2[0])) + ((p1[1] - p2[1])*(p1[1] - p2[1]))))


if __name__=="__main__":
    # bar_plot_mod()
    # loss_plot()
    if sys.argv[1] == "tsne":
        vector_path = sys.argv[2]
        label_path = sys.argv[3]

        vector = np.load(open(vector_path,"rb"))
        label = np.load(open(label_path,"rb"))

        plot_embedding_tsne(vector, label)

        # Calculate centroids
        tsne_feature_array = np.load(open('./logs/tsne/tsne_feat.npy',"rb"))

        pmed, nmed = calculate_centroids(tsne_feature_array, label)
        dist = calculate_distance(pmed, nmed)

        print(dist)

