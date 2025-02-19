import shutil
import random
import requests
import zipfile, tarfile, rarfile
import os
from tqdm import tqdm
from io import BytesIO

from huggingface_hub import snapshot_download
from datasets import load_dataset

class Task:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.get_question()
    def get_question(self):
        if hasattr(self, "problem"):
            self.question = self.problem
        elif hasattr(self, "question"):
            self.question = self.question
        elif hasattr(self, "puzzle"):
            self.question = self.puzzle
        else:
            self.question = None
    


class EvalDataset:
    def __init__(self, name, save_path = "evaldatasets"):
        self.name = name
        self.save_path = save_path
        self.dataset_path = os.path.join(self.save_path,self.name)
        self.support_datasets = ["MATH", "MATH-500", "GSM8K", "CommonsenseQA", "ECQA", "Omni-MATH", "MMLU-Pro", "ZebraLogicBench", "ZebraLogicBench-private"]
        assert self.name in self.support_datasets, f'"{self.name}" currently are not supported. The supported datasets: {self.support_datasets}'
        self.point = "test"
        self.train_datasets = []
        self.test_datasets = []
        self.valid_datasets = []
        # 下载数据集
        self.download()
        # 处理下载的数据集
        self.process()


# =================================================================
 
    def download(self,name = None):
        if name is not None:
            self.name = name

        if os.path.exists(self.dataset_path):
            print("Dataset already exists.")
            return 
        if self.name == "MATH":
            # self.url_download("https://people.eecs.berkeley.edu/~hendrycks/MATH.tar", self.save_path)
            self.huggingface_download("hendrycks/competition_math")
            os.rename(f"{self.save_path}/hendrycks___competition_math",f"{self.save_path}/MATH")
        if self.name == "MATH-500":
            # self.url_download("https://people.eecs.berkeley.edu/~hendrycks/MATH.tar", self.save_path)
            self.huggingface_download("HuggingFaceH4/MATH-500")
            os.rename(f"{self.save_path}/HuggingFaceH4___MATH-500",f"{self.save_path}/MATH-500")
   
        if self.name == "GSM8K":
            self.huggingface_download("openai/gsm8k")
            os.rename(f"{self.save_path}/openai___gsm8k",f"{self.save_path}/GSM8K")

        if self.name == "CommonsenseQA": 
            self.huggingface_download("tau/commonsense_qa")
            os.rename(f"{self.save_path}/tau___commonsense_qa",f"{self.save_path}/{self.name}")    
        
        if self.name == "ECQA":
            self.huggingface_download("tasksource/ecqa") 
            os.rename(f"{self.save_path}/tasksource___ecqa",f"{self.save_path}/{self.name}")    
        
        if self.name == "Omni-MATH":
            self.huggingface_download("KbsdJames/Omni-MATH") 
            os.rename(f"{self.save_path}/KbsdJames___omni-math",f"{self.save_path}/{self.name}")    
        
        if self.name == "MMLU-Pro":
            self.huggingface_download("TIGER-Lab/MMLU-Pro") 
            os.rename(f"{self.save_path}/TIGER-Lab___mmlu-pro",f"{self.save_path}/{self.name}")

        if self.name == "ZebraLogicBench":
            self.huggingface_download("allenai/ZebraLogicBench") 
            os.rename(f"{self.save_path}/allenai___zebra_logic_bench",f"{self.save_path}/{self.name}")
        
        if self.name == "ZebraLogicBench-private":
            self.huggingface_download("allenai/ZebraLogicBench-private") 
            os.rename(f"{self.save_path}/allenai___zebra_logic_bench-private",f"{self.save_path}/{self.name}")
         

    
    def url_download(self, url, save_path="evaldatasets", dataset_name="Dataset"):
        # 如果保存路径不存在，则创建它
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 获取文件扩展名以判断压缩格式，并获取文件名（不包括查询参数）
        file_name = url.split('/')[-1].split('?')[0]
        file_extension = os.path.splitext(file_name)[-1].lower()
        file_path = os.path.join(save_path, file_name)
        extract_dir = os.path.join(save_path, os.path.splitext(file_name)[0])  # 解压后的文件夹路径

        support_extension = ['.zip', '.tar', '.gz', '.bz2', '.rar']
        assert file_extension in support_extension, f"File extension {file_extension} is not supported."

        # 检查解压后的文件夹是否已经存在于本地
        if os.path.exists(extract_dir) and os.path.isdir(extract_dir):
            print(f"The extracted directory '{extract_dir}' already exists. Checking for completeness...")
            
            # 这里可以添加更详细的检查逻辑，例如检查某些关键文件是否存在或验证文件完整性。
            # 由于这可能需要具体了解文件结构，这里我们简单假设文件夹存在即为完整。
            print("The extracted files are considered complete. Skipping download.")
            return

        try:
            # 发送 HTTP 请求并获取响应对象
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 检查请求是否成功
        except Exception as e:
            print(f"An error occurred while extracting the archive: {e}")
            

        # 获取文件总大小
        total_size = int(response.headers.get('content-length', 0))

        # 初始化进度条，并设置描述（即进度条的名字）
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dataset_name) as progress_bar:
            content = BytesIO()
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                content.write(data)
            progress_bar.close()

        # 确保所有数据都已写入 BytesIO 对象
        content.seek(0)
        print("Zipping ...")
        # 根据文件扩展名解压文件到指定路径
        try:
            if file_extension == '.zip':
                with zipfile.ZipFile(content, 'r') as zip_ref:
                    zip_ref.extractall(save_path)
            elif file_extension in ['.tar', '.gz', '.bz2']:
                with tarfile.open(fileobj=content, mode='r:*') as tar_ref:
                    tar_ref.extractall(path=save_path)
            elif file_extension == '.rar':
                with rarfile.RarFile(content) as rar_ref:
                    rar_ref.extractall(path=save_path)
            else:
                raise ValueError(f"Download Failed for the url: {url}")
        except Exception as e:
            print(f"An error occurred while extracting the archive: {e}")
            if os.path.exists(extract_dir):  # 如果解压失败，删除部分解压的文件夹
                shutil.rmtree(extract_dir)
            raise
        print("Zipped ...")

    def huggingface_download(self,repo_id):

        try:
            if self.name in ["MATH", "GSM8K"]:
                branch = "main"
            elif self.name in ["ZebraLogicBench","ZebraLogicBench-private"]:
                branch = "mc_mode"
            else:
                branch = "default"
            load_dataset(repo_id, branch, cache_dir = self.save_path, trust_remote_code=True) 

        except Exception as e:
            print(e)
            # 如果是没连上网，则切换成镜像网站
            print("Changed to url: https://hf-mirror.com. Retrying...")
            os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
            try:
                load_dataset(repo_id, "main", cache_dir = self.save_path, trust_remote_code=True)
            except Exception as e:
                print(e)
        
            
# =================================================================
    def process(self):
        DATASET = load_dataset(self.dataset_path)
        print(DATASET)
        if "train" in DATASET.keys():
            self.point = "train"
            train_ds = DATASET['train']
            train_ds.map(self.deal)
        if "test" in DATASET.keys(): 
            self.point = "test"
            test_ds = DATASET['test']
            test_ds.map(self.deal)
        else:
            self.point = "valid"
            test_ds = DATASET['validation']
            test_ds.map(self.deal)
        
    def deal(self,example):

        task = Task(**example)
        if self.point == "test":
            self.test_datasets.append(task)
        else:
            self.train_datasets.append(task)

    
 

# ==================================================================
     
    def show(self):
        # 展示任务的组成结构
        try:
            print(self.test_datasets[0].__dict__.keys())
        except  Exception as e:
            print(e)


    def get_datasets(self, num:str|list|int = "all", point = "test"):
        datasets = []
        if point == "train":
            datasets = self.train_datasets
        if point == "test":
            datasets  =  self.test_datasets
        if point == "valid":
            datasets =  self.valid_datasets
        # print(datasets)
        try:
            if num == "all":
                return datasets
            elif isinstance(num, list):
                return datasets[num[0]:num[1]]
            else:
                return random.choice(datasets)
        except Exception as e:
            print(e)

 
if __name__ == '__main__':
 
    dataset = EvalDataset("ZebraLogicBench-private")
    dataset.show()
    # print(dataset.train_datasets[0].__dict__)
    print(dataset.get_datasets([0,1]))
