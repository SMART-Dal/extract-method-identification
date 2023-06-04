from transformers import  AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import json, torch, tensor
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from ..embeddings.bert_based import Bert
import numpy as np
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

class AE:

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
    
    
    def __get_data_from_jsonl(path):

        data = []
        with open(path, 'r') as file:
            for line in file:
                item = json.loads(line)
                if len(item['positive_case_methods'])==0:
                    continue
                data.append(item)
        
        return data

    def data_generator(data, batch_size):
        num_samples = len(data)
        steps_per_epoch = num_samples // batch_size

        while True:
            for i in range(steps_per_epoch):
            # for i in range(1):
                batch = data[i * batch_size: (i + 1) * batch_size]

                merged_embeddings = []
                for positive_embeddings, negative_embeddings in Bert.embedding_generator(batch):
                    merged_embeddings += (positive_embeddings + negative_embeddings)

                yield np.asarray(merged_embeddings), np.asarray(merged_embeddings)
    
    def save_checkpoint(self):
        return ModelCheckpoint(
                    'model_checkpoint.h5',
                    monitor='loss',  # Monitor the loss value
                    save_freq='epoch',  # Save checkpoints after each epoch
                    save_best_only=True  # Save only the best model
                )
    def model_train(self):
        data = self.__get_data_from_jsonl('/content/file_0000.jsonl') #Change it

        autoencoder = self.__get_autoencoder(768,32)

        autoencoder.compile(optimizer='adam', loss='mse')
        csv_logger_callback = CSVLogger('loss_curves.csv')
        early_stopping_callback = EarlyStopping(patience=3, restore_best_weights=True)

        model_checkpoint_callback = self.save_checkpoint()
        # autoencoder.fit(self.data_generator(data,8), epochs=50, steps_per_epoch=len(data) // 8)
        autoencoder.fit(self.data_generator(data,8), epochs=50, steps_per_epoch=len(data) // 8,
                        callbacks=[model_checkpoint_callback, csv_logger_callback, early_stopping_callback])

if __name__=="__main__":
    AE.model_train()


