import os
import sys
import torch
from RoBERTa_Custom.fine_tune import main

def check_gpu_environment():
    """檢查 GPU 環境並返回相關信息"""
    gpu_info = {
        "is_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "NA",
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NA"
    }
    
    print("=== GPU 環境信息 ===")
    print(f"CUDA 是否可用: {gpu_info['is_available']}")
    print(f"GPU 數量: {gpu_info['gpu_count']}")
    print(f"CUDA 版本: {gpu_info['cuda_version']}")
    print(f"GPU 型號: {gpu_info['gpu_model']}")
    print("==================")
    
    return gpu_info

if __name__ == "__main__":
    gpu_info = check_gpu_environment()
    
    if not gpu_info['is_available']:
        print("警告: 未檢測到可用的 GPU，訓練過程可能會很慢")
        user_input = input("是否繼續運行？(y/n): ")
        if user_input.lower() != 'y':
            print("程序已終止")
            sys.exit(0)
    
    # 設置 GPU 記憶體管理
    if gpu_info['is_available']:
        torch.cuda.empty_cache()  # 清空 GPU 快取
        torch.backends.cudnn.benchmark = True  # 啟用 cudnn 自動優化
    
    print("\n開始執行微調訓練...")
    main()