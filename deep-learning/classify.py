import json
import numpy as np

def embeddings_generator(file_path, batch_size):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Calculate the number of batches
        num_batches = len(lines) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_data = lines[start_idx:end_idx]
            embeddings_list = []
            labels = []

            for line in batch_data:
                json_data = json.loads(line)
                pos_methods = json_data['pos_methods']
                neg_methods = json_data['neg_methods']

                # Call your function to get the embeddings from pos_methods
                embeddings = get_embeddings(pos_methods)

                embeddings_list.append(embeddings)
                labels.append(1)  # Positive label

                # You can also process the embeddings from neg_methods if needed
                # neg_embeddings = get_embeddings(neg_methods)
                # embeddings_list.append(neg_embeddings)
                # labels.append(0)  # Negative label

            yield np.array(embeddings_list), np.array(labels)
