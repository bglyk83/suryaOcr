import os
import json
from PIL import Image
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr

# 테스트 이미지 선택
test_image_path = "./data/images/insure_00001.jpeg"
test_image = Image.open(test_image_path).convert("RGB")

print(f"테스트 이미지: {test_image_path}")
print(f"이미지 크기: {test_image.size}")

# 모델 로드
det_model = load_det_model()
det_processor = load_det_processor()
rec_model = load_rec_model()
rec_processor = load_rec_processor()

print("모델 로드 완료")

# OCR 실행
images = [test_image]
languages = [["ko"]]  # 한국어로 설정

print("OCR 실행 중...")
results = run_ocr(images, languages, det_model, det_processor, rec_model, rec_processor)

# 결과 출력
for i, result in enumerate(results):
    print(f"\n=== 이미지 {i+1} OCR 결과 ===")
    for j, text_line in enumerate(result.text_lines):
        print(f"텍스트 {j+1}: {text_line.text}")
        print(f"  위치: {text_line.bbox}")
        print(f"  신뢰도: {text_line.confidence:.3f}")

print("\nOCR 테스트 완료!")
