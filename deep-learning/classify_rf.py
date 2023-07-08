from sklearn.ensemble import RandomForestClassifier
import torch, numpy as np, json, pickle, time, os, sys
from tqdm import tqdm
from bert_based import Bert
from autoencoder_pn import Autoencoder
from metrics import get_metrics_classicalml
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report



def load_autoencoder_model(model_path, n_inputs, encoding_dim, device):
    autoencoder = Autoencoder(n_inputs, encoding_dim)
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder

def get_bottleneck_representation(em, input_dim, encoding_dim):

    model_path = "./trained_models/autoencoder_gc_pn_128_150.pth"  # Path to the saved model
    n_inputs = input_dim  # Input dimension
    encoding_dim = encoding_dim  # Latent space dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = load_autoencoder_model(model_path, n_inputs, encoding_dim, device)
    with torch.no_grad():
        bottleneck_representation = autoencoder.encoder(em)
    return bottleneck_representation

def train_rf_ae(train_data, train_label, using_AE=True):

    bert = Bert("microsoft/graphcodebert-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in train_data]

    batch_size = 8
    num_samples = len(tokenized_data)
    bottleneck_rep_list = []

    print("Tokenization Done. Num of Samples - ", num_samples)

    for i in tqdm(range(0, num_samples, batch_size)):

        batch_tokenized_data = tokenized_data[i:i+batch_size]

        input_ids = torch.tensor(batch_tokenized_data).to(device)

        with torch.cuda.amp.autocast():
            batch_embeddings = bert.generate_embeddings(input_ids)

        bottleneck_rep = get_bottleneck_representation(batch_embeddings,768,128)

        # bottleneck_rep_list.append(batch_embeddings.cpu())
        bottleneck_rep_list.append(bottleneck_rep.cpu())

    bottleneck_rep_arr = np.concatenate(bottleneck_rep_list, axis=0)

    print("Label shape - ", train_label.shape)
    print("Input shape - ", bottleneck_rep_arr.shape)

    # Save the embedding Vector
    os.makedirs('./logs/vectors/',exist_ok=True)
    np.save('./logs/vectors/ae_encoded_vector.npy',bottleneck_rep_arr)

    rf = RandomForestClassifier(100,n_jobs=-1, random_state=42)
    lr = LogisticRegression(max_iter=1000,random_state=42)
    dt = DecisionTreeClassifier(random_state=42)

    rf_fit = rf.fit(bottleneck_rep_arr,train_label)
    lr_fit = lr.fit(bottleneck_rep_arr,train_label)
    dt_fit = dt.fit(bottleneck_rep_arr,train_label)

    print("Done Fitting")

    rf_score = cross_val_score(rf, bottleneck_rep_arr,train_label, cv=5, scoring="accuracy")
    lr_score = cross_val_score(lr, bottleneck_rep_arr,train_label, cv=5, scoring="accuracy")
    dt_score = cross_val_score(dt, bottleneck_rep_arr,train_label, cv=5, scoring="accuracy")

    print("Mean score for rf - ", rf_score.mean())
    print("Mean score for lr - ", lr_score.mean())
    print("Mean score for dt - ", dt_score.mean())

    #Grid Search

    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    start_time = time.time()
    grid_search.fit(bottleneck_rep_arr,train_label)
    print("GS Time - ", time.time()-start_time)
    print("Best Params - ", grid_search.best_params_)

    random_forest = RandomForestClassifier(max_depth=80, max_features=2, min_samples_leaf=3,min_samples_split=10,n_estimators=1000)
    random_forest.fit(bottleneck_rep_arr,train_label)

    return random_forest

def test_rf_ae(test_data, test_label, model):

    bert = Bert("microsoft/graphcodebert-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenized_data = [bert.tokenizer.encode(text, padding='max_length', truncation=True, max_length=512) for text in test_data]

    batch_size = 8
    num_samples = len(tokenized_data)
    bottleneck_rep_list = []

    print("Tokenization Done. Num of Samples - ", num_samples)

    for i in tqdm(range(0, num_samples, batch_size)):

        batch_tokenized_data = tokenized_data[i:i+batch_size]

        input_ids = torch.tensor(batch_tokenized_data).to(device)

        with torch.cuda.amp.autocast():
            batch_embeddings = bert.generate_embeddings(input_ids)

        bottleneck_rep = get_bottleneck_representation(batch_embeddings,768,128)

        # bottleneck_rep_list.append(batch_embeddings.cpu())
        bottleneck_rep_list.append(bottleneck_rep.cpu())

    bottleneck_rep_arr = np.concatenate(bottleneck_rep_list, axis=0)

    print("Label shape - ", test_label.shape)
    print("Input shape - ", bottleneck_rep_arr.shape)

    pred_label = model.predict(bottleneck_rep_arr)

    ac,pr,re,f1 = get_metrics_classicalml(test_label, pred_label)

    print("Accuracy - ",round(ac,3))
    print("Precision - ",round(pr,3))
    print("Recall - ",round(re,3))
    print("F-1 - ",round(f1,3))

    print(classification_report(test_label,pred_label))



if __name__=="__main__":


    train_data_file_path = sys.argv[1]
    test_data_file_path = sys.argv[2]
    train_label_file_path = sys.argv[3]
    test_label_file_path = sys.argv[4]


    with open(train_data_file_path,"+rb") as f:
        train_data_arr = np.load(f)

    with open(train_label_file_path,"+rb") as f:
        train_label_arr = np.load(f)

    print("Training Data Shape - ",train_data_arr.shape)
    print("Training Label Shape - ",train_label_arr.shape)
    rf_model = train_rf_ae(train_data_arr, train_label_arr)

    rf_model_path = "./trained_models/Classification/RandomForest"
    os.makedirs(rf_model_path, exist_ok=True)
    
    with open (os.path.join(rf_model_path,"rf_ae.pkl"),"wb") as f:
        pickle.dump(rf_model,f)

    with open(test_data_file_path,"+rb") as f:
        test_data_arr = np.load(f)
    with open(test_label_file_path,"+rb") as f:
        test_label_arr = np.load(f)

    with open(os.path.join(rf_model_path,"rf_ae.pkl"),"rb") as f:
        model = pickle.load(f)

    print("Testing...")
    test_rf_ae(test_data_arr,test_label_arr, rf_model)



