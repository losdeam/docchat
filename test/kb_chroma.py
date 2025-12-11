from rag.retriever import Chroma_Builder
import traceback
retriever = Chroma_Builder()
try:
    print(retriever.config)
    # retriever.add_doc(["test/data/2510.18234v1.pdf"])
    print(retriever.invoke("deepseek"))
except Exception as e:
    traceback.print_exc()
finally:
    retriever.save_local()
