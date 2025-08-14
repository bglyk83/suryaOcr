import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

DATA_DIR = "./data"
IMG_DIR = os.path.join(DATA_DIR, "images")
LABEL_DIR = os.path.join(DATA_DIR, "labels")

rec_records = []

for label_file in os.listdir(LABEL_DIR):
    with open(os.path.join(LABEL_DIR, label_file), "r", encoding="utf-8") as f:
        data = json.load(f)

    # 이미지 파일명 추출 (JSON 구조에 맞게 조정)
    img_name = data.get("name") or data.get("image") or label_file.replace(".json", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.isfile(img_path):
        print(f"Image file not found: {img_path}")
        continue

    image = Image.open(img_path).convert("RGB")

    # 실제 데이터 구조에 맞게 수정
    for ann in data["annotations"]:
        for poly in ann.get("polygons", []):
            # text는 polygon 객체 내부에 있음
            text = poly.get("text", "")
            if not text:  # 빈 텍스트는 건너뛰기
                continue
                
            # polygon의 points를 사용하여 bbox 계산
            points = poly["points"]
            if len(points) < 4:
                continue
                
            # polygon을 bbox로 변환
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            # 이미지 크기 내에서 클리핑
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(image.width, int(x_max))
            y_max = min(image.height, int(y_max))
            
            # 유효한 bbox인지 확인
            if x_max > x_min and y_max > y_min:
                crop = image.crop((x_min, y_min, x_max, y_max))
                rec_records.append({"image": crop, "label": text})

print(f"총 {len(rec_records)}개의 텍스트 샘플을 찾았습니다.")

# 모델 / 프로세서 로드
rec_model = load_rec_model()
rec_processor = load_rec_processor()

# Dataset 클래스 정의
class OCRDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        # 이미지 PIL -> numpy
        image_np = np.array(item["image"])
        
        # Surya의 실제 사용법에 맞게 processor 호출
        encoding = self.processor(
            images=image_np, 
            text=item["label"], 
            lang=["ko"],  # 한국어를 리스트로 설정
            return_tensors="pt", 
            padding=True
        )
        
        # encoding은 딕셔너리 형태로 반환됨
        return encoding

def collate_fn(batch):
    """배치 처리를 위한 커스텀 collate 함수"""
    # 배치의 모든 키를 수집
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    # 각 키에 대해 배치 처리
    batch_dict = {}
    for key in all_keys:
        if key in ["pixel_values", "labels", "langs"]:
            # 텐서인 경우 스택
            tensors = [item[key] for item in batch if key in item]
            if tensors:
                # 시퀀스 길이가 다른 경우 패딩
                if key == "labels":
                    max_len = max(tensor.shape[-1] for tensor in tensors)
                    padded_tensors = []
                    for tensor in tensors:
                        if tensor.shape[-1] < max_len:
                            # 패딩 토큰으로 패딩
                            pad_size = max_len - tensor.shape[-1]
                            padded = torch.cat([tensor, torch.zeros(tensor.shape[0], pad_size, dtype=tensor.dtype)], dim=-1)
                            padded_tensors.append(padded)
                        else:
                            padded_tensors.append(tensor)
                    batch_dict[key] = torch.cat(padded_tensors, dim=0)
                else:
                    batch_dict[key] = torch.cat(tensors, dim=0)
    
    return batch_dict

train_dataset = OCRDataset(rec_records, rec_processor)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rec_model.to(device)
rec_model.train()

optimizer = torch.optim.AdamW(rec_model.parameters(), lr=3e-5)

# 학습 루프
for epoch in range(5):
    epoch_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # 배치를 디바이스로 이동
        inputs = {k: v.to(device) for k, v in batch.items()}
        
        # 모델 forward pass
        outputs = rec_model(**inputs)
        
        # Loss 계산 (labels가 있는 경우)
        if "labels" in inputs:
            loss = outputs.loss
        else:
            # labels가 없는 경우 logits와 labels를 직접 계산
            logits = outputs.logits
            labels = inputs.get("labels")
            if labels is not None:
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(train_loader):.4f}")

print("학습 완료!")
