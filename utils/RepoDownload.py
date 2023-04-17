import pygit2,tempfile, os, uuid, shutil, stat, subprocess
from utils.file_folder_remover import Remover
class CloneRepo:

    BASE_URL = "https://github.com/"

    def __init__(self, repo_name, repo_url) -> None:
        self.repo_name = repo_name
        self.repo_url = repo_url

        

    def clone_repo(self):

        uuid = str(uuid.uuid4())

        clone_path = os.path.join(tempfile.gettempdir(),"auto-ref", self.repo_name)

        tdir_clone = clone_path
        if os.path.exists(tdir_clone):
            shutil.rmtree(tdir_clone, onerror=lambda func, path, _: (
                os.chmod(path, stat.S_IWRITE), func(path)))
        os.makedirs(tdir_clone)
        try:
            pygit2.clone_repository(self.repo_url+".git",tdir_clone)
            # subprocess.call(["git","clone",self.repo_url+".git",tdir_clone])
            return tdir_clone
        except Exception as e:
            # print("Error Cloning Repository - ",self.repo_name)
            Remover(tdir_clone).remove_folder()
            print(e)
            raise Exception("Error Cloning Repository - ",self.repo_name)

