import pydriller, json
from utils.file_folder_remover import Remover

class MethodExtractor:

    def __init__(self,repo_path,json_path) -> None:
        self.repo_path = repo_path
        self.json_path = json_path
        self.pos_methods, self.neg_methods = [],[]

    # This function is used to parse the output json file generated by RefactoringMiner
    def json_parser(self):

        with open(self.json_path) as f:
            json_output = json.load(f)
            
        dict_output = {}

        '''
        The output json is structured as follows:

        {
            "commits":[{REFs applied in this commit along with commit metadata}]
        }

        Each object inside the commits array is structured as follows:

        {
            "repository": "https://github.com/danilofes/refactoring-toy-example.git",
            "sha1": "9a9878aeb62a6bb6ff2bed6c03dd1dd7ed1f202b",
            "url": "https://github.com/danilofes/refactoring-toy-example/commit/9a9878aeb62a6bb6ff2bed6c03dd1dd7ed1f202b",
            "refactorings": [{Details of all types of refactorings applied in this commit}]
        }

        Each object inside the refactorings array is structured as follows:

        {
                "type": "Pull Up Method",
                "description": "Pull Up Method public meow() : void from class org.felines.Cat to public meow() : void from class org.felines.Feline",
                "leftSideLocations": [{
                    "filePath": "src/org/felines/Cat.java",
                    "startLine": 5,
                    "endLine": 7,
                    "startColumn": 2,
                    "endColumn": 3,
                    "codeElementType": "METHOD_DECLARATION",
                    "description": "original method declaration",
                    "codeElement": "public meow() : void"
            }],
                "rightSideLocations": [{
                    "filePath": "src/org/felines/Feline.java",
                    "startLine": 5,
                    "endLine": 7,
                    "startColumn": 2,
                    "endColumn": 3,
                    "codeElementType": "METHOD_DECLARATION",
                    "description": "pulled up method declaration",
                    "codeElement": "public meow() : void"
            }]
        }           

        '''

        #Logic for ExtractMethod refactoring 
        for obj in json_output["commits"]:
            if len(obj["refactorings"])==0:
                continue
            extract_method_list = list(filter(lambda x: x["type"]=="Extract Method",obj["refactorings"]))
            if len(extract_method_list)==0:
                continue
            extract_method_details_in_commit = {
                "pos_method":[],
                "neg_method":[]                
            }
            for em_obj in extract_method_list:

                # Take only the refactoring metadata which has the source code line numbers before and after refactoring
                to_be_refactored_obj = list(filter(lambda x: x["codeElementType"]=="METHOD_DECLARATION" and x["description"]=="source method declaration before extraction",em_obj["leftSideLocations"]))[0]
                refactored_obj = list(filter(lambda x: x["codeElementType"]=="METHOD_DECLARATION" and x["description"]=="source method declaration after extraction",em_obj["rightSideLocations"]))[0]

                extract_method_details_in_commit["pos_method"].append(to_be_refactored_obj)
                extract_method_details_in_commit["neg_method"].append(refactored_obj)
                
            dict_output[obj["sha1"]]=extract_method_details_in_commit

        return dict_output
    
    def extract_method_body(self, parsed_json_dict:dict):
        
        for commit in pydriller.Repository(self.repo_path,only_commits=list(parsed_json_dict.keys())).traverse_commits():

            modified_file_list  = commit.modified_files

            len_pos_methods = len(self.pos_methods)
            
            file_paths = [x.new_path for x in modified_file_list]
            for pos_id,meta_data in enumerate(parsed_json_dict[commit.hash]["neg_method"]):
                mod_file_index = file_paths.index(meta_data["filePath"])
                mod_file = modified_file_list[mod_file_index]
            
                self.pos_methods.append(self.__split_and_extract_methods(mod_file.source_code_before,parsed_json_dict[commit.hash]["pos_method"][pos_id]["startLine"],parsed_json_dict[commit.hash]["pos_method"][pos_id]["endLine"]))
                self.neg_methods.append(self.__split_and_extract_methods(mod_file.source_code,meta_data["startLine"],meta_data["endLine"]))

                if len(self.pos_methods)>len_pos_methods:
                    print((len(self.pos_methods)-len_pos_methods)/len(mod_file.methods_before))
                    

        return self.pos_methods, self.neg_methods

    def __split_and_extract_methods(self, source_code,start_line, end_line):
        source_code_lines = str(source_code).splitlines()
        return "\n".join(source_code_lines[start_line-1:end_line])


if __name__=="__main__":
    json_parser = MethodExtractor(repo_path="/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/refactoring-toy-example",json_path="/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/refactoring_identifier/output/output.json")
    parsed_json_dict = json_parser.json_parser()
    pos, neg = json_parser.extract_method_body(parsed_json_dict)
    print(len(pos),len(neg))