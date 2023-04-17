import pydriller, json
from utils.file_folder_remover import Remover

class MethodExtractor:

    def __init__(self,repo_path,json_path) -> None:
        self.repo_path = repo_path
        self.json_path = json_path
        self.pos_methods, self.neg_methods = [],[]

    def json_parser(self):
        #TODO
        #Filter for Method Refs
        with open(self.json_path) as f:
            json_output = json.load(f)
            

        dict_output = {}
        for obj in json_output["commits"]:
            if len(obj["refactorings"])==0:
                continue
            extract_method_list = list(filter(lambda x: x["type"]=="Extract Method",obj["refactorings"]))
            for em_obj in extract_method_list:
                to_be_refactored_obj = list(filter(lambda x: x["codeElementType"]=="METHOD_DECLARATION",em_obj["leftSideLocations"]))[0]
                refactored_obj = list(filter(lambda x: x["codeElementType"]=="METHOD_DECLARATION" and x["description"]=="source method declaration after extraction",em_obj["rightSideLocations"]))[0]
                dict_output[obj["sha1"]]={
                    "pos_method":to_be_refactored_obj,
                    "neg_method":refactored_obj
                }

        return dict_output
    
    def extract_method_body(self, parsed_json_dict:dict):
        
        for commit in pydriller.Repository(self.repo_path,only_commits=list(parsed_json_dict.keys())).traverse_commits():
            # mod_files = list(filter(lambda x: x.filename==,commit.modified_files))
            for mod_file in commit.modified_files:

                self.pos_methods.append(self.__split_and_extract_methods(mod_file.source_code_before,parsed_json_dict[commit.hash]["pos_method"]["startLine"],parsed_json_dict[commit.hash]["pos_method"]["endLine"]))
                self.neg_methods.append(self.__split_and_extract_methods(mod_file.source_code_before,parsed_json_dict[commit.hash]["pos_method"]["startLine"],parsed_json_dict[commit.hash]["pos_method"]["endLine"]))
        
        return self.pos_methods, self.neg_methods


    def __split_and_extract_methods(self, source_code,start_line, end_line):
        source_code_lines = str(source_code).splitlines()
        return "\n".join(source_code_lines[start_line-1:end_line])


if __name__=="__main__":
    json_parser = MethodExtractor(repo_path="/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/refactoring-toy-example",json_path="/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/refactoring_identifier/output/output.json")
    # print(json_parser.json_parser().keys())
    parsed_json_dict = json_parser.json_parser()
    pos, neg = json_parser.extract_method_body(parsed_json_dict)
    print(len(pos),len(neg))