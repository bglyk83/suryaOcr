import os
import json
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from surya.model.detection.segformer import load_model, load_processor

import wandb
wandb.init(mode="disabled")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_DIR = "./data/images"
LABEL_DIR = "./data/labels"

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [x_min, y_min, x_max, y_max]

def create_segmentation_map(annotations, image_size, target_size):
    """annotation을 segmentation map으로 변환"""
    seg_map = np.zeros(target_size, dtype=np.uint8)
    
    # 원본 이미지 크기에서 target 크기로의 스케일 계산
    orig_h, orig_w = image_size
    target_h, target_w = target_size
    
    scale_w = target_w / orig_w
    scale_h = target_h / orig_h
    
    for ann in annotations:
        for poly in ann.get("polygons", []):
            points = poly["points"]
            if len(points) < 4:
                continue
            
            # polygon을 bbox로 변환
            bbox = polygon_to_bbox(points)
            x_min, y_min, x_max, y_max = bbox
            
            # 스케일링
            x_min = int(x_min * scale_w)
            y_min = int(y_min * scale_h)
            x_max = int(x_max * scale_w)
            y_max = int(y_max * scale_h)
            
            # 경계 내로 클리핑
            x_min = max(0, min(x_min, target_w - 1))
            y_min = max(0, min(y_min, target_h - 1))
            x_max = max(0, min(x_max, target_w - 1))
            y_max = max(0, min(y_max, target_h - 1))
            
            # segmentation map에 텍스트 영역 표시
            if x_max > x_min and y_max > y_min:
                cv2.fillPoly(seg_map, [np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])], 1)
    
    return seg_map

class DetDataset(Dataset):
    def __init__(self, label_dir, img_dir, processor, size=(384, 384)):
        self.records = []
        self.processor = processor
        self.size = size

        for label_file in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_name = data.get("name") or data.get("image") or label_file.replace(".json", ".jpg")
            img_path = os.path.join(img_dir, img_name)
            if not os.path.isfile(img_path):
                print(f"Image file does NOT exist: {img_path}")
                continue

            image = Image.open(img_path).convert("RGB")
            orig_size = image.size[::-1]  # (height, width)
            image = image.resize(self.size)
            image_np = np.array(image)

            # segmentation map 생성
            seg_map = create_segmentation_map(data["annotations"], orig_size, self.size)

            self.records.append({
                "image": image_np,
                "segmentation_map": seg_map
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]

        # Surya processor는 이미지만 받음
        encoding = self.processor(
            images=item["image"],
            return_tensors="pt"
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # segmentation map을 tensor로 변환
        seg_map = torch.from_numpy(item["segmentation_map"]).long()
        encoding["labels"] = seg_map

        return encoding

def collate_fn(batch):
    # 딕셔너리 리스트 -> 딕셔너리로 묶기
    return {
        key: torch.stack([b[key] for b in batch])
        for key in batch[0]
    }

def main():
    print("=== 개선된 Detection 모델 학습 ===\n")
    
    det_processor = load_processor()
    det_model = load_model()
    det_model.to(device)
    det_model.train()

    train_dataset = DetDataset(LABEL_DIR, IMG_DIR, det_processor, size=(384, 384))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 더 낮은 학습률과 더 긴 학습 시간
    optimizer = torch.optim.AdamW(det_model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 모델 저장 디렉토리 생성
    os.makedirs("./det_model_improved", exist_ok=True)
    
    best_loss = float('inf')
    patience = 0
    max_patience = 3

    print(f"총 {len(train_dataset)}개 이미지로 학습 시작")
    print(f"배치 크기: 2, 학습률: 1e-5")
    print(f"디바이스: {device}\n")

    for epoch in range(20):  # 더 많은 에포크
        epoch_loss = 0
        det_model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = det_model(**inputs)

            # 실제 segmentation map을 사용하여 loss 계산
            logits = outputs.logits
            labels = inputs["labels"]
            
            # 모델 출력 크기에 맞게 labels 리사이즈
            batch_size, num_classes, height, width = logits.shape
            labels_resized = torch.nn.functional.interpolate(
                labels.unsqueeze(1).float(), 
                size=(height, width), 
                mode='nearest'
            ).squeeze(1).long()
            
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels_resized)

            optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(det_model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        print(f"Epoch {epoch + 1} 평균 Loss: {avg_loss:.4f}")
        print(f"현재 학습률: {scheduler.get_last_lr()[0]:.2e}")
        
        # 최고 성능 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            print(f"  🎉 새로운 최고 성능! Loss: {best_loss:.4f}")
            print(f"  💾 모델 저장 중...")
            
            # 모델 상태 저장
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': det_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, './det_model_improved/best_model.pth')
            
            # 모델 전체 저장 (추론용)
            det_model.save_pretrained('./det_model_improved')
            det_processor.save_pretrained('./det_model_improved')
            print(f"  ✅ 모델 저장 완료: ./det_model_improved/")
        else:
            patience += 1
            print(f"  ⏳ 성능 개선 없음 (patience: {patience}/{max_patience})")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        print()

    print(f"\n🎯 학습 완료!")
    print(f"📊 최고 성능 Loss: {best_loss:.4f}")
    print(f"💾 저장된 모델: ./det_model_improved/")
    
    # 최종 모델 저장
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': det_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
    }, './det_model_improved/final_model.pth')

if __name__ == "__main__":
    main() 