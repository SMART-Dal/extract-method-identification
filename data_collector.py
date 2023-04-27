import csv,time, sys, json, os
from multiprocessing import Pool, Lock
from pymongo import MongoClient
from utils.RepoDownload import CloneRepo
from refactoring_identifier.RefMiner import RefMiner
from utils.file_folder_remover import Remover
from utils.method_code_extractor import MethodExtractor
from embeddings.bert_based import Bert
from utils.Database import Database, PostgresDatabase
from joblib import Parallel, delayed
from dotenv import dotenv_values


# Define the number of workers to use for parallel processing
NUM_WORKERS = 4

#Lock
lock = Lock()

# Function to clone a Git repo, run a Java program, and parse the output
def process_repo(repo_details):

    
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


        # DB
        # pg_db = PostgresDatabase()
        # pg_db.cursor.execute("INSERT INTO repodetails (repo_name, repo_url) VALUES (%s, %s)", (db_dict["repo_name"], db_dict["repo_url"]))
        # pg_db.connection.commit()
        # repo_table_id = pg_db.cursor.fetchone()[0]

        # for pos_method_code, neeg_method_code in zip(db_dict["positive_case_methods"], db_dict["negative_case_methods"]):
        #     pg_db.cursor.execute("INSERT INTO sourcecodedata (repo_details_fkey, pos_source_code, neg_source_code) VALUES (%s, %s, %s)", (repo_table_id, pos_method_code, neeg_method_code))
        #     pg_db.connection.commit()

        # pg_db.connection.commit()
        # pg_db.cursor.close()
        # pg_db.connection.close()


        # Locally in jsonl 
        out_jsonl_path = os.path.join(os.getcwd(),"data","output")

        if not os.path.exists(out_jsonl_path):
            os.mkdir(out_jsonl_path)
        
        with open(os.path.join(out_jsonl_path,dotenv_values(".env")["output_file_name"]), 'a') as f: # os.environ["output_file_name"] doesn't work in MP
            f.write(json.dumps(db_dict) + "\n")
    
    # Generate Embeddings
    # try:
    #     gc_embedding_object = Bert("microsoft/graphcodebert-base")
    #     pos_gc_embeddings = [gc_embedding_object.generate_individual_embedding(pos_method_body) for pos_method_body in pos_method_body_list]
    #     neg_gc_embeddings = [gc_embedding_object.generate_individual_embedding(neg_method_body) for neg_method_body in neg_method_body_list]

    #     db_dict = {
    #         "repo_name":repo_details[0],
    #         "repo_url": repo_details[1],
    #         "positive_case_methods":pos_method_body_list,
    #         "negative_case_methods":neg_method_body_list,
    #         "positive_case_embedding": pos_gc_embeddings,
    #         "negative_case_embedding":neg_gc_embeddings
    #     }

    #     gc_db.insert_doc(db_dict)
    #     print(f"DB updated with GC for {repo_details[0]}")
    # except Exception as e:
    #     print(f"Error creating/updating graph code bert embeddings for repository - {repo_details[0]}")
    #     return


    # try:

    #     cb_embedding_object = Bert("microsoft/codebert-base")
    #     pos_cb_embeddings = [cb_embedding_object.generate_individual_embedding(pos_method_body) for pos_method_body in pos_method_body_list]
    #     neg_cb_embeddings = [cb_embedding_object.generate_individual_embedding(neg_method_body) for neg_method_body in neg_method_body_list]

    #     db_dict = {
    #         "repo_name":repo_details[0],
    #         "repo_url": repo_details[1],
    #         "positive_case_methods":pos_method_body_list,
    #         "negative_case_methods":neg_method_body_list,            
    #         "positive_case_embedding": pos_cb_embeddings,
    #         "negative_case_embedding":neg_cb_embeddings
    #     }

    #     cb_db.insert_doc(db_dict)
    #     print(f"DB updated with CB for {repo_details[0]}")
    # except Exception as e:
    #     print(f"Error creating/updating code bert embeddings for repository - {repo_details[0]}")
    #     return

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
    # input_file = sys.argv[1]
    input_file = "/home/ip1102/Ref-Res/extract-method-identification/data/test.csv"

    with open(input_file,"r") as f:
        reader = csv.reader(f)
        repo_details = [(row[0],row[1]) for row in reader]

    t = time.time()

    # Use multiprocessing to process the repos in parallel
    with Pool(NUM_WORKERS) as p:
        p.map(process_repo, repo_details)


    # #Use Joblib
    # Parallel(n_jobs=NUM_WORKERS)(delayed(process_repo)(repo_detail) for repo_detail in repo_details)

    print(time.time()-t)

