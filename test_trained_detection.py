import os
import json
import numpy as np
from PIL import Image
import torch
from surya.model.detection.segformer import load_model, load_processor
from surya.detection import batch_text_detection

def test_trained_detection():
    """학습된 detection 모델 테스트"""
    print("=== 학습된 Detection 모델 테스트 ===\n")
    
    # 테스트 이미지 선택
    test_image_path = "./data/images/insure_00001.jpeg"
    test_image = Image.open(test_image_path).convert("RGB")
    
    print(f"테스트 이미지: {test_image_path}")
    print(f"이미지 크기: {test_image.size}")
    
    # 원본 모델과 학습된 모델 로드
    print("\n1. 원본 모델 로드...")
    original_model = load_model()
    original_processor = load_processor()
    
    print("2. 학습된 모델 로드...")
    trained_model = load_model(checkpoint="./det_model")
    trained_processor = load_processor()
    
    print("모델 로드 완료!\n")
    
    # 원본 모델로 테스트
    print("=== 원본 모델 테스트 ===")
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start_time:
        start_time.record()
    
    original_results = batch_text_detection(
        [test_image], 
        original_model, 
        original_processor
    )
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        original_time = start_time.elapsed_time(end_time) / 1000.0
    else:
        original_time = 0
    
    original_bboxes = [line.bbox for line in original_results[0].bboxes]
    print(f"원본 모델 감지된 텍스트 영역: {len(original_bboxes)}개")
    print(f"원본 모델 처리시간: {original_time:.2f}초")
    
    # 학습된 모델로 테스트
    print("\n=== 학습된 모델 테스트 ===")
    if start_time:
        start_time.record()
    
    trained_results = batch_text_detection(
        [test_image], 
        trained_model, 
        trained_processor
    )
    
    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        trained_time = start_time.elapsed_time(end_time) / 1000.0
    else:
        trained_time = 0
    
    trained_bboxes = [line.bbox for line in trained_results[0].bboxes]
    print(f"학습된 모델 감지된 텍스트 영역: {len(trained_bboxes)}개")
    print(f"학습된 모델 처리시간: {trained_time:.2f}초")
    
    # 성능 비교
    print("\n=== 성능 비교 ===")
    print(f"감지된 영역 수:")
    print(f"  원본 모델: {len(original_bboxes)}개")
    print(f"  학습된 모델: {len(trained_bboxes)}개")
    print(f"  차이: {len(trained_bboxes) - len(original_bboxes)}개")
    
    print(f"\n처리 시간:")
    print(f"  원본 모델: {original_time:.2f}초")
    print(f"  학습된 모델: {trained_time:.2f}초")
    if original_time > 0:
        speed_ratio = trained_time / original_time
        print(f"  속도 비율: {speed_ratio:.2f}x")
    
    # Ground truth와 비교
    print("\n=== Ground Truth 비교 ===")
    label_file = "insure_00001.json"
    label_path = os.path.join("./data/labels", label_file)
    
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        gt_count = 0
        for ann in data["annotations"]:
            gt_count += len(ann.get("polygons", []))
        
        print(f"Ground Truth 텍스트 영역: {gt_count}개")
        print(f"원본 모델 정확도: {len(original_bboxes)/gt_count*100:.1f}%" if gt_count > 0 else "계산 불가")
        print(f"학습된 모델 정확도: {len(trained_bboxes)/gt_count*100:.1f}%" if gt_count > 0 else "계산 불가")
    
    # 상세 결과 출력
    print("\n=== 상세 결과 ===")
    print("원본 모델 감지 영역 (처음 5개):")
    for i, bbox in enumerate(original_bboxes[:5]):
        print(f"  {i+1}: {bbox}")
    
    print("\n학습된 모델 감지 영역 (처음 5개):")
    for i, bbox in enumerate(trained_bboxes[:5]):
        print(f"  {i+1}: {bbox}")
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    test_trained_detection() 