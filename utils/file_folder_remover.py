import os, subprocess

class Remover:

    def __init__(self, path) -> None:
        self.path = path

    def remove_folder(self):
        try:
            if os.path.exists(self.path):
                os.rmdir(self.path)
                return True
            else:
                print(f"The path {self.path} does not exist!")
                return False
        except Exception as e:
            out = subprocess.run(["rm","-rf",self.path])
            try:
                out.check_returncode()
            except subprocess.CalledProcessError as error:
                print(error)
                return False
            
    def remove_file(self):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
                return True
            else:
                print(f"The path {self.path} does not exist!")
                return False
        except Exception as e:
            print(e)
            return False

if __name__=="__main__":
    Remover("/home/ip1102/projects/def-tusharma/ip1102/Ref-Res/Research/embeddings/code2vec.py").remove_file()