import torch
from torchvision import models, transforms
from PIL import Image

# 加載預訓練的 ResNet 模型
model = models.resnet50(pretrained=True)
# 設置模型為評估模式，對於推斷（測試）階段很重要
model.eval()

# 定義圖像預處理轉換
preprocess = transforms.Compose([
    # 將圖像的短邊調整為 256 像素，保持長寬比
    transforms.Resize(256),
    # 從圖像中心裁剪出一個 224x224 的區域，這是 ResNet 模型預期的輸入大小
    transforms.CenterCrop(224),
    # 將圖像轉換為 Tensor 格式，這是 PyTorch 中處理圖像的標準格式
    transforms.ToTensor(),
    # 正規化圖像的像素值，目的是使圖像的像素值在接近 0 的範圍內波動，有助模型的穩定訓練和更快速的收斂
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 載入圖像路徑
img_path = "./dog.jpg"
# 讀取圖像
img = Image.open(img_path)

# 使用定義好的轉換對圖像進行預處理
img_t = preprocess(img)
# 在第一維增加一個維度，準備作為模型的輸入批次
batch_t = torch.unsqueeze(img_t, 0)

# 進行預測，避免梯度計算
with torch.no_grad():
    # 使用模型進行預測
    out = model(batch_t)

# 讀取 ImageNet 類別標籤
with open("imagenet_classes.txt") as f:
    # 讀取類別標籤
    labels = [line.strip() for line in f.readlines()]

# 找出最大機率的類別索引
_, index = torch.max(out, 1)
# 計算機率並轉換為百分比
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# 輸出預測類別和其機率百分比
print(labels[index[0]], percentage[index[0]].item())
