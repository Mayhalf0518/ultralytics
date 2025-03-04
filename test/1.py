import torch

if torch.cuda.is_available():
    print("✅ GPU 已啟用:", torch.cuda.get_device_name(0))
else:
    print("❌ 沒有使用 GPU，正在使用 CPU")
