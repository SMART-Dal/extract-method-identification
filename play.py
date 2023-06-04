import multiprocessing, subprocess
import urllib.request

class PipelineHealth:

    def __init__(self) -> None:
        print("No. of cores - ",multiprocessing.cpu_count())
        print("Internet Connection - ",self.check_internet_connection())
        print("Check Java Version - ", self.check_java())

    def check_internet_connection(self):
        try:
            urllib.request.urlopen("https://www.google.com/")
            return True
        except Exception as e:
            print(e)
            return False
        
    def check_java(self):
        subprocess_output = subprocess.run(["java","-version"],capture_output=True)
        print("Output - ", subprocess_output.stdout)
        print("Error - ",subprocess_output.stderr)



if __name__=="__main__":
    list1 = [1,2,3]
    list2 = [4,5,6]
    lst = list1+list2
    print(lst)
    # PipelineHealth()
