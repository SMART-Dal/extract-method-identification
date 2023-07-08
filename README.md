# Automatic Refactoring Candidate Identification Leveraging Effective Code Representation

## Steps to reproduce

### Dependencies and Environment Set-Up

- Clone this repository to the local folder

- `cd` into the project directory

- Create a virtual environment using `python -m venv <venv_name>`

- Activate the virtualenv by `source <venv_name>/bin/activate`

- Install all the dependencies by `pip install -r requirements.txt`

- The code is optimized for GPU (cuda) so running it on a GPU with decent memory should work fine. If there is memory issue, please lower the batch size.


### Extract Method Dataset Creation

- The file input.csv contains all the repository names from the baseline for which the dataset is to be created. 

- To  split the file in 5 parts, execute the following - `source csv-splitter.sh`. The split size can be configured from the script.

- We then took 5% of the repositories, from first 2 splits to extract the positive and negative samples. 

- To generate the positive and negative samples, execute the following - `python data_creator.py <inpute_file_path> <output_file_path>`

- This will create the .jsonl file with the identified samples. These files are already generated and present in https://doi.org/10.5281/zenodo.8122619


### Deep Learning

- First navigate to the deep learning folder using - `cd deep-learning`

- Splitting the dataset (.jsonl file) to train and test split:

- To split the dataset execute the following - `python data_creator.py <input_file_path> <output_file_name>`. It will split the dataset and store it as .npy files as `data/np_arrays/output_file_name.npz`

- To train and test the autoencoder model, execute the following - `python autoencoder_pn.py <train_data_file_path> <test_data_file_path>` 

- This will store the validated model in `./trained_models/AE/models`

- To train and test the random forest classifier, execute the following - `python classify_rf.py <train_data_file_path> <train_label_file_path> <test_data_file_path> <test_label_file_path>`

- It will generate the encoded embeddings on the fly and the final evaluation metrics will be displayed on the terminal. 

- To plot the t-SNE plots, execute the `cd` into `deep-learning/metrics/` and execute the following - `python plot.py tsne <embedded_representation_path> <label_array_path>`


