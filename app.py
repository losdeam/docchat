from pages import page_init
import atexit
import signal,sys,os
from rag.retriever import kb_manager
import threading

class CleanupManager:
    """清理管理器，防止重复执行"""
    def __init__(self):
        self._cleanup_called = False
        self._lock = threading.Lock()
    
    def cleanup(self):
        """清理函数，确保只执行一次"""
        with self._lock:
            if self._cleanup_called:
                print("清理已执行过，跳过")
                return False
            
            self._cleanup_called = True
            print("执行清理操作...")
            
            # 你的清理代码
            try:
                kb_manager.raise_()
            except Exception as e:
                print(f"清理过程中出错: {e}")
            
            return True
    
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n接收到退出信号 {signum}")
        
        # 执行清理
        cleaned = self.cleanup()
        
        if cleaned:
            print("清理完成，退出程序")
            # 注意：这里不调用 sys.exit()，让 atexit 处理
            # 或者直接退出，不触发 atexit
            os._exit(0)  # 快速退出，不触发 atexit
        else:
            print("清理已执行过，直接退出")
            os._exit(0)
    
    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """异常处理器"""
        print(f"发生未捕获的异常: {exc_type.__name__}: {exc_value}")
        
        # 执行清理
        self.cleanup()
        
        # 调用原始异常处理器
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def setup_hooks(self):
        """设置钩子"""
        # 1. 注册 atexit
        atexit.register(self.cleanup)
        
        # 2. 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 3. 设置异常钩子
        sys.excepthook = self.exception_hook

# 全局清理管理器
cleanup_manager = CleanupManager()

if __name__ == "__main__":
    cleanup_manager.setup_hooks()
    page_init()