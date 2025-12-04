from utils import file_manager

file_manage = file_manager()
print(file_manage.file_cache_path)
print(file_manage.key_json)


print(file_manage.add_docs(["test/data/2510.18234v1.pdf"]))
file_manage.save_json()