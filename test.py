import torch

def check_gpu_status():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
        
        # 測試 GPU 運算
        x = torch.rand(5, 3)
        print("\n測試 GPU 張量運算:")
        print("CPU 張量:", x)
        x_gpu = x.cuda()
        print("GPU 張量:", x_gpu)
        print("裝置位置:", x_gpu.device)
    else:
        print("警告: CUDA 不可用，請檢查 PyTorch 安裝和 NVIDIA 驅動程式")

if __name__ == "__main__":
    check_gpu_status()