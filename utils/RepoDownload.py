import tempfile, os, uuid, shutil, stat, subprocess
from utils.file_folder_remover import Remover

class CloneRepo:

    BASE_URL = "https://github.com/"

    def __init__(self, repo_name, repo_url) -> None:
        self.repo_name = repo_name
        self.repo_url = repo_url

        

    def clone_repo(self):

        clone_path = os.path.join(tempfile.gettempdir(),"auto-ref", self.repo_name)

        tdir_clone = clone_path
        if os.path.exists(tdir_clone):
            shutil.rmtree(tdir_clone, onerror=lambda func, path, _: (
                os.chmod(path, stat.S_IWRITE), func(path)))
        os.makedirs(tdir_clone)
        try:
            clone_output = subprocess.run(["git","clone",self.repo_url+".git",tdir_clone],capture_output=True)
            clone_output.check_returncode()
            return tdir_clone
        except Exception as e:
            Remover(tdir_clone).remove_folder()
            print(e)
            raise Exception("Error Cloning Repository - ",self.repo_name)

