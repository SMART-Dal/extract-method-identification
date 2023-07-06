import csv,time, sys, json, os, concurrent.futures
from multiprocessing import Pool, Lock
from pymongo import MongoClient
from utils.RepoDownload import CloneRepo
from refactoring_identifier.RefMiner import RefMiner
from utils.file_folder_remover import Remover
from utils.method_code_extractor import MethodExtractor
from embeddings.bert_based import Bert
from utils.Database import Database
from joblib import Parallel, delayed
from dotenv import dotenv_values


# Define the number of workers to use for parallel processing
NUM_WORKERS = 6

#Lock
lock = Lock()

# Function to clone a Git repo, run a Java program, and parse the output
def process_repo(repo_details):

    print("Start analyzing for repo - "+repo_details[0])
    #Clone the repo
    try:
        cloned_path = CloneRepo(repo_details[0], repo_details[1]).clone_repo()
    except Exception as e:
        print(e)
        return

    #Run RefactoringMiner
    try:
        ref_output_path = RefMiner().exec_refactoring_miner(cloned_path,repo_details[0])
    except Exception as e:
        print(e)
        return   

    #Prase the Json and extract the methods
    try:
        me_obj = MethodExtractor(cloned_path,ref_output_path)
        parsed_json_dict = me_obj.json_parser()
        pos_method_body_list, neg_method_body_list = me_obj.extract_method_body(parsed_json_dict)
        Remover(cloned_path).remove_folder()
        Remover(ref_output_path).remove_file()
    except Exception as e:
        print(f"Error extracting positive and negative methods for {repo_details[0]}")
        print(e)
        return
    
    #Establish DB Connections
    # gc_db = Database("GraphCodeBert_DB")
    # gc_db = Database("test_cc")
    # cb_db = Database("CodeBert_DB")

    #Store the methods
    db_dict = {
        "repo_name":repo_details[0],
        "repo_url": repo_details[1],
        "positive_case_methods":pos_method_body_list,
        "negative_case_methods":neg_method_body_list
    }

    
    with lock:


        # Locally in jsonl 
        out_jsonl_path = os.path.join(os.environ['SLURM_TMPDIR'],'extract-method-identification',"data","output")

        if not os.path.exists(out_jsonl_path):
            os.mkdir(out_jsonl_path)


        with open(os.path.join(out_jsonl_path,output_file_name), 'a') as f: # os.environ["output_file_name"] doesn't work in MP
            f.write(json.dumps(db_dict) + "\n")
            f.flush()
            f.close()
    
    print("End analysis for repo - "+repo_details[0])


def run_process(NUM_WORKERS, process_repo):
    with open("/home/ip1102/Playground/multi_tp/data/input.csv", "r") as f:
        reader = csv.reader(f)
        repo_details = [(row[0],row[1]) for row in reader]
    t = time.time()

    # Use multiprocessing to process the repos in parallel
    # with Pool(NUM_WORKERS) as p:
    #     p.map(process_repo, repo_details)

    #Use Joblib
    Parallel(n_jobs=NUM_WORKERS)(delayed(process_repo)(repo_detail) for repo_detail in repo_details)
    return t

if __name__=="__main__":

    # print(sys.argv[1])
    input_file = sys.argv[1]
    output_file_name = sys.argv[2]
    # input_file = "/home/ip1102/Ref-Res/extract-method-identification/data/test.csv"

    with open(input_file,"r") as f:
        reader = csv.reader(f)
        repo_details = [(row[0],row[1]) for row in reader]

    t = time.time()

    # Use multiprocessing to process the repos in parallel
    # with Pool(NUM_WORKERS) as p:
    #     p.map(process_repo, repo_details)

    #Use MultiThread
    with concurrent.futures.ThreadPoolExecutor(NUM_WORKERS) as executor:
        executor.map(process_repo, repo_details)


    # #Use Joblib
    # Parallel(n_jobs=NUM_WORKERS)(delayed(process_repo)(repo_detail) for repo_detail in repo_details)

    print(time.time()-t)

