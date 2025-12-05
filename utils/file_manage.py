import hashlib
from config.settings import settings
from typing import Any, List
import logging,os,json
class file_manager:
    """
    文件管理器
    定义文件管理器的标准接口
    """
    def __init__(self):
        """
        初始化文件管理器
        参数:
            docs: 文件路径列表
        """
        
        # 首先获取整体的文件缓存
        self.file_cache_path = settings.DOC_CACHE_PATH
        self.load_local()
        self.docs_hashes = frozenset(self.key_json.keys()) if self.key_json else frozenset() # 存储文档的集合，用户快速查重

    def load_local(self)->List[str]:
        """
        从本地缓存的参数文件加载基础信息
        """
        # 获取路径
        key_json_path = os.path.join(self.file_cache_path, settings.KEY_JSON_NAME)
        # 如果文件不存在，则创建一个
        if not os.path.exists(key_json_path): 
            os.makedirs(self.file_cache_path, exist_ok=True)# 保证父目录存在
            with open(key_json_path, "w") as f:
                json.dump({}, f)
            self.key_json = {}
        else:
            # 读取json文件为dict格式
            self.key_json = json.load(open(key_json_path, "r"))

    def add_docs(self,file_path_list:List[str]):
        """
        向知识库中添加文档
        参数:
            file_path: 待加入的文档路径列表
        """
        add_file = set()
        for file_path in file_path_list:
            with open(file_path, "rb") as f:
                file_name = os.path.basename(file_path)
                file_hash = hashlib.sha256(f.read()).hexdigest() 
                hash_name = file_hash+"."+file_path.split(".")[-1] #获取文件的哈希值与后缀名
                if hash_name not in self.docs_hashes:
                    # 向缓存地址中写入文档
                    with open(os.path.join(self.file_cache_path, hash_name), "wb") as f_cache:
                        f_cache.write(f.read())
                    self.key_json[hash_name] = file_name
                    add_file.add(file_path)
        if not add_file:
            return "上传的文件列表为空或是文件均已存在"
        activate_set = set(self.docs_hashes)
        activate_set.update(add_file)
        self.docs_hashes = frozenset(activate_set)
        return "文件上传成功"
    def save_json(self):
        """保存参数文件"""
        key_json_path = os.path.join(self.file_cache_path, settings.KEY_JSON_NAME)
        with open(key_json_path, "w") as f:
            json.dump(self.key_json, f)

    # def _get_file_hashes() -> frozenset:
    #     """Generate SHA-256 hashes for uploaded files."""
    #     hashes = {}
    #     for file in uploaded_files:
    #         with open(file.name, "rb") as f:
    #             hashes[hashlib.sha256(f.read()).hexdigest()] = f.read()
    #     return frozenset(hashes)
file_manager_activate = file_manager()
def get_single_hash(file_path:str)->str:
    """Generate SHA-256 hash for a single file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()