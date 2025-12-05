from utils import file_manager_activate


print(file_manager_activate.file_cache_path)
print(file_manager_activate.key_json)


print(file_manager_activate.add_docs(["test/data/2510.18234v1.pdf"]))
file_manager_activate.save_json()