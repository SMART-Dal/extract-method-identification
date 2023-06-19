import json, sys, os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from bert_based import Bert
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sklearn.utils import shuffle
from tqdm import tqdm
class AE:

    def __init__(self, path) -> None:
        self.data_file = path
        self.bert = Bert("microsoft/graphcodebert-base")

    def __get_autoencoder(self,n_inputs,latent_space_dim=32):

        input_shape = (n_inputs,)
        encoding_dim = latent_space_dim

        input_layer = Input(shape=input_shape)
        encoder = Dense(256, activation='relu')(input_layer)
        encoder = Dense(128, activation='relu')(encoder)
        encoder = Dense(64, activation='relu')(encoder)
        latent_space = Dense(encoding_dim, activation='relu')(encoder)
        decoder = Dense(64, activation='relu')(latent_space)
        decoder = Dense(128, activation='relu')(decoder)
        decoder = Dense(256, activation='relu')(decoder)
        output_layer = Dense(n_inputs, activation='sigmoid')(decoder)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)

        return autoencoder
    
    
    def __get_data_from_jsonl(self):

        # data = []
        # with open(self.data_file, 'r') as file:
        #     for line in file:
        #         item = json.loads(line)
        #         if len(item['positive_case_methods'])==0:
        #             continue
        #         data.append(item)
        
        # return data

        data = []
        max_p = 0
        with open(self.data_file, 'r') as file:
            for line in file:
                item = json.loads(line)
                if len(item['positive_case_methods'])==0:
                    continue
                if len(item['positive_case_methods'])>max_p:
                    max_p=len(item['positive_case_methods'])
                data+=item['positive_case_methods']

        return data

    def data_generator(self,data, batch_size):
        num_samples = len(data)
        steps_per_epoch = num_samples // batch_size

        # while True:
        for i in range(steps_per_epoch):
        # for i in range(1):
            batch = data[i * batch_size: (i + 1) * batch_size]

            merged_embeddings = []
            for positive_embeddings, negative_embeddings in self.bert.embedding_generator(batch):
                merged_embeddings += (positive_embeddings + negative_embeddings)

            # yield np.asarray(merged_embeddings), np.asarray(merged_embeddings)
            return merged_embeddings, merged_embeddings
    
    def save_checkpoint(self):
        checkpoint_filepath = 'autoencoder_checkpoint.h5'
        return ModelCheckpoint(
                    checkpoint_filepath,
                    save_weights_only=False,
                    save_freq='epoch'
                )

    def fetch_batch(self,X, batch_size, batch):
        start = batch*batch_size
        X_batch = X[start:start+batch_size]
        X_batch = self.bert.gen_embeddings(X_batch)
        X_batch = np.asarray(X_batch)
        return X_batch, X_batch

    
    def model_train(self):

        print("Start training")

        print("Parse jsonl file for training")
        data = self.__get_data_from_jsonl() #Change it


        print("Get autoencoder model")
        autoencoder = self.__get_autoencoder(768,32)

        batch_size = 16
        loss_history = []
        val_loss_history = []
        acc_history = []
        val_acc_history = []
        n_epochs = 10
        

        autoencoder.compile(optimizer='adam', loss='mse')
        csv_logger_callback = CSVLogger('loss_curves.csv')
        early_stopping_callback = EarlyStopping(patience=3, restore_best_weights=True)

        model_checkpoint_callback = self.save_checkpoint()

        for epoch in range(n_epochs):
            X = shuffle(data, random_state = epoch**2)
            for batch in tqdm(range(len(X) //batch_size)):
            
                X_batch, y_batch = self.fetch_batch(X, batch_size, batch)
                # X_batch, y_batch = fetch_random_batch(X, y, batch_size)
                loss = autoencoder.train_on_batch(X_batch, y_batch)
            
            loss_history.append(loss)
            # acc_history.append(acc)
            
            # Run validtion at the end of each epoch.
            # y_pred = model1.predict(X_test)
            # val_loss, val_acc = model1.evaluate(X_test, y_test)
            # val_loss_history.append(val_loss)
            # val_acc_history.append(val_acc)
                
            print(f"Epoch: {epoch+1}, Loss: {loss}")
            model_checkpoint_callback.on_epoch_end(epoch=epoch,logs={'loss':loss})
            # print('Epoch: %d, Train Loss %.3f, Train Acc. %.3f' %
            #         (epoch+1, loss, acc))

        # autoencoder.fit(self.data_generator(data,8), epochs=50, steps_per_epoch=len(data) // 8)

        # print("Fit model")
        # autoencoder.fit(self.data_generator(data,8), epochs=15, steps_per_epoch=len(data) // 8,
        #                 callbacks=[model_checkpoint_callback, csv_logger_callback, early_stopping_callback])

if __name__=="__main__":
    # input_file = sys.argv[1]
    input_file = os.path.join(os.environ['SLURM_TMPDIR'],'file_0000.jsonl')
    AE(input_file).model_train()


