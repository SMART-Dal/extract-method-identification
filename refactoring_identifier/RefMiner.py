import os, subprocess

# print(os.environ['refresearch'])

class RefMiner:

    def __init__(self) -> None:
        self.ref_bin_path = os.path.join(os.environ['refresearch'],"refactoring_identifier","executable","RefactoringMiner","bin")
        self.cwd = os.getcwd()
        self.output_path = os.path.join(os.environ['refresearch'],"refactoring_identifier","output")
    def exec_refactoring_miner(self, repo_path,repo_name):
        try:
            output_path = os.path.join(self.output_path,repo_name+".json")
            os.chdir(self.ref_bin_path)
            ref_exec = subprocess.run(["sh","RefactoringMiner","-a",repo_path,"-json",output_path],capture_output=True)
            ref_exec.check_returncode()
            os.chdir(self.cwd)
            return output_path

        except subprocess.CalledProcessError as error:
            print(ref_exec.stdout)
            print(error)
            os.chdir(self.cwd)
            raise Exception("Error running RefactoringMiner in repository - ",repo_path)
        except Exception as e:
            print(e)
            os.chdir(self.cwd)
            raise Exception("Error running RefactoringMiner in repository - ",repo_path)


if __name__=="__main__":
    RefMiner().exec_refactoring_miner("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/refactoring-toy-example","ref_toy")
