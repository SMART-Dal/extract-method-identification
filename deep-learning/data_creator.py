from sklearn.model_selection import train_test_split
import numpy as np, json, random, pickle, sys


def __get_data_from_jsonl(data_file):

    data, labels = [], []
    max_p, empty_repo = 0, 0
    with open(data_file, 'r') as file:
        for line in file:
            item = json.loads(line)
            if len(item['positive_case_methods'])==0:
                empty_repo+=1
                continue 
            if len(item['positive_case_methods'])>max_p:
                max_p=len(item['positive_case_methods'])

            # if len(data)>=10000:
            #     break            
            
            data+=item['positive_case_methods']

            labels.extend([1]*len(item['positive_case_methods']))
            data+=item['negative_case_methods']

            labels.extend([0]*len(item['positive_case_methods']))
        try:
            assert len(labels)==len(data)
        except AssertionError as e:
            print(len(labels))
            print(len(data))
    print("Total samples - ", len(data))
    print("Maximum methods per case in a repo - ", max_p)
    print("Empty Repositories - ",empty_repo)
    return data, labels

def get_train_test_val_split(data, labels):
    
    train_data, test_data, train_label, test_label = train_test_split(data,labels, test_size=0.2, stratify=labels)
    print(f"Training sample length - {len(train_data)}. Validation Sample length - {len(test_data)}")
    print(f"Training label length - {len(train_label)}. Validation label length - {len(test_label)}")
    return train_data, test_data, train_label, test_label

def split_by_ratio(test_data, test_labels, ratio=0.85):

    combined_data = list(zip(test_data, test_labels))

    print("Test set splitting based on identified distribution....")
    ones_indices = [i for i, label in enumerate(test_labels) if label == 1]
    num_ones = len(ones_indices)
    num_to_remove = int(ratio * num_ones)

    random.shuffle(ones_indices)
    indices_to_remove = ones_indices[:num_to_remove]

    filtered_data = []
    filtered_labels = []

    for i, (data, label) in enumerate(combined_data):
        if i not in indices_to_remove or label == 0:
            filtered_data.append(data)
            filtered_labels.append(label)
    
    return filtered_data, filtered_labels

def save_nps(file_path, file_name, halfsize=True, split_by_size=True):

    data, labels = __get_data_from_jsonl(file_path)
    
    if halfsize:
        data = data[:len(data)//2]
        labels = labels[:len(labels)//2]
    
    train_data, test_data, train_label, test_label = get_train_test_val_split(data,labels)

    train_data_np = np.asarray(train_data)
    train_label_np = np.asarray(train_label)

    if split_by_size:
        test_data,test_label = split_by_ratio(test_data, test_label)

    test_data_np = np.asarray(test_data)
    test_label_np = np.asarray(test_label)

    print("Train Data Shape",train_data_np.shape)
    print("Train Label Shape", train_label_np.shape)
    
    print("Test Data Shape",test_data_np.shape)
    print("Test Label Shape", test_label_np.shape)

    with open ("../data/np_arrays/train_data_"+file_name+".npy","+wb") as f:
        np.save(f,train_data_np)

    with open ("../data/np_arrays/test_data_"+file_name+".npy","+wb") as f:
        np.save(f,test_data_np)

    with open ("../data/np_arrays/train_label_"+file_name+".npy","+wb") as f:
        np.save(f,train_label_np)

    with open ("../data/np_arrays/test_label_"+file_name+".npy","+wb") as f:
        np.save(f,test_label_np)

if __name__=="__main__":

    print("Start...")
    input_file_name = sys.argv[1]
    print(input_file_name)
    # save_nps(f"../data/output/{input_file_name}",input_file_name,
    #          halfsize=True,
    #          split_by_size=True)