import os
import json
import time
from PIL import Image
import numpy as np
from collections import defaultdict
import re
from surya.model.detection.segformer import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr

def calculate_cer(pred, true):
    """Character Error Rate 계산"""
    if not true:
        return 1.0 if pred else 0.0
    
    # 한글, 영문, 숫자만 추출
    pred_clean = re.sub(r'[^\w\s가-힣]', '', pred.lower())
    true_clean = re.sub(r'[^\w\s가-힣]', '', true.lower())
    
    if not true_clean:
        return 1.0 if pred_clean else 0.0
    
    # Levenshtein distance 계산
    m, n = len(true_clean), len(pred_clean)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if true_clean[i-1] == pred_clean[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / len(true_clean)

def calculate_wer(pred, true):
    """Word Error Rate 계산"""
    if not true:
        return 1.0 if pred else 0.0
    
    # 단어로 분리
    pred_words = re.findall(r'\w+', pred.lower())
    true_words = re.findall(r'\w+', true.lower())
    
    if not true_words:
        return 1.0 if pred_words else 0.0
    
    # Levenshtein distance 계산
    m, n = len(true_words), len(pred_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if true_words[i-1] == pred_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n] / len(true_words)

def extract_ground_truth(label_path):
    """JSON 파일에서 ground truth 텍스트 추출"""
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    for ann in data["annotations"]:
        for poly in ann.get("polygons", []):
            text = poly.get("text", "")
            if text.strip():
                texts.append(text.strip())
    
    return texts

def benchmark_ocr():
    """OCR 성능 벤치마크 실행"""
    print("=== Surya OCR 성능 벤치마크 ===\n")
    
    # 모델 로드
    print("모델 로딩 중...")
    start_time = time.time()
    det_model = load_det_model()
    det_processor = load_det_processor()
    rec_model = load_rec_model()
    rec_processor = load_rec_processor()
    model_load_time = time.time() - start_time
    print(f"모델 로드 완료: {model_load_time:.2f}초\n")
    
    # 데이터 준비
    DATA_DIR = "./data"
    IMG_DIR = os.path.join(DATA_DIR, "images")
    LABEL_DIR = os.path.join(DATA_DIR, "labels")
    
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"총 {len(image_files)}개 이미지 테스트\n")
    
    # 성능 지표 초기화
    total_cer = 0
    total_wer = 0
    total_texts = 0
    total_detection_time = 0
    total_recognition_time = 0
    total_processing_time = 0
    confidence_scores = []
    
    results_by_image = []
    
    for i, img_file in enumerate(image_files):
        print(f"처리 중: {img_file} ({i+1}/{len(image_files)})")
        
        # 이미지 로드
        img_path = os.path.join(IMG_DIR, img_file)
        image = Image.open(img_path).convert("RGB")
        
        # Ground truth 로드
        label_file = img_file.rsplit('.', 1)[0] + '.json'
        label_path = os.path.join(LABEL_DIR, label_file)
        
        if not os.path.exists(label_path):
            print(f"  ⚠️  라벨 파일 없음: {label_file}")
            continue
        
        ground_truth_texts = extract_ground_truth(label_path)
        
        # OCR 실행
        start_time = time.time()
        images = [image]
        languages = [["ko"]]
        
        results = run_ocr(images, languages, det_model, det_processor, rec_model, rec_processor)
        processing_time = time.time() - start_time
        
        # 결과 분석
        predicted_texts = []
        confidences = []
        
        for result in results:
            for text_line in result.text_lines:
                predicted_texts.append(text_line.text)
                if text_line.confidence is not None and not np.isnan(text_line.confidence):
                    confidences.append(text_line.confidence)
        
        # 성능 지표 계산
        image_cer = 0
        image_wer = 0
        matched_count = 0
        
        # 텍스트 매칭 (간단한 방법)
        for pred_text in predicted_texts:
            best_cer = float('inf')
            best_wer = float('inf')
            
            for gt_text in ground_truth_texts:
                cer = calculate_cer(pred_text, gt_text)
                wer = calculate_wer(pred_text, gt_text)
                
                if cer < best_cer:
                    best_cer = cer
                if wer < best_wer:
                    best_wer = wer
            
            if best_cer < 0.5:  # 50% 이하의 CER을 매칭으로 간주
                matched_count += 1
            
            image_cer += best_cer
            image_wer += best_wer
        
        if predicted_texts:
            image_cer /= len(predicted_texts)
            image_wer /= len(predicted_texts)
        
        # 통계 업데이트
        total_cer += image_cer * len(predicted_texts)
        total_wer += image_wer * len(predicted_texts)
        total_texts += len(predicted_texts)
        total_processing_time += processing_time
        confidence_scores.extend(confidences)
        
        # 이미지별 결과 저장
        results_by_image.append({
            'image': img_file,
            'predicted_count': len(predicted_texts),
            'ground_truth_count': len(ground_truth_texts),
            'cer': image_cer,
            'wer': image_wer,
            'processing_time': processing_time,
            'avg_confidence': np.mean(confidences) if confidences else 0
        })
        
        print(f"  📊 예측: {len(predicted_texts)}개, 실제: {len(ground_truth_texts)}개")
        print(f"  📈 CER: {image_cer:.3f}, WER: {image_wer:.3f}")
        print(f"  ⏱️  처리시간: {processing_time:.2f}초")
        print(f"  🎯 평균 신뢰도: {np.mean(confidences) if confidences else 0:.3f}\n")
    
    # 전체 성능 지표 계산
    if total_texts > 0:
        avg_cer = total_cer / total_texts
        avg_wer = total_wer / total_texts
    else:
        avg_cer = avg_wer = 0
    
    avg_processing_time = total_processing_time / len(image_files)
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    # 결과 출력
    print("=" * 50)
    print(" 전체 성능 지표")
    print("=" * 50)
    print(f"  처리된 이미지: {len(image_files)}개")
    print(f" 총 텍스트: {total_texts}개")
    print(f"  총 처리시간: {total_processing_time:.2f}초")
    print(f" 평균 처리시간: {avg_processing_time:.2f}초/이미지")
    print(f" 처리속도: {len(image_files)/total_processing_time:.2f} 이미지/초")
    print()
    print(f" 평균 CER: {avg_cer:.3f}")
    print(f" 평균 WER: {avg_wer:.3f}")
    print(f" 평균 신뢰도: {avg_confidence:.3f}")
    print(f" 신뢰도 표준편차: {np.std(confidence_scores) if confidence_scores else 0:.3f}")
    print()
    
    # 이미지별 상세 결과
    print(" 이미지별 상세 결과")
    print("-" * 50)
    for result in results_by_image:
        print(f"{result['image']:20} | "
              f"예측: {result['predicted_count']:3d} | "
              f"실제: {result['ground_truth_count']:3d} | "
              f"CER: {result['cer']:.3f} | "
              f"WER: {result['wer']:.3f} | "
              f"시간: {result['processing_time']:.2f}s | "
              f"신뢰도: {result['avg_confidence']:.3f}")
    
    # 성능 등급 평가
    print("\n 성능 등급 평가")
    print("-" * 30)
    
    if avg_cer < 0.1:
        cer_grade = "A+ (우수)"
    elif avg_cer < 0.2:
        cer_grade = "A (양호)"
    elif avg_cer < 0.3:
        cer_grade = "B (보통)"
    elif avg_cer < 0.5:
        cer_grade = "C (미흡)"
    else:
        cer_grade = "D (불량)"
    
    if avg_wer < 0.2:
        wer_grade = "A+ (우수)"
    elif avg_wer < 0.4:
        wer_grade = "A (양호)"
    elif avg_wer < 0.6:
        wer_grade = "B (보통)"
    elif avg_wer < 0.8:
        wer_grade = "C (미흡)"
    else:
        wer_grade = "D (불량)"
    
    print(f" CER 등급: {cer_grade}")
    print(f" WER 등급: {wer_grade}")
    print(f" 속도 등급: {'빠름' if avg_processing_time < 10 else '보통' if avg_processing_time < 30 else '느림'}")
    
    return {
        'avg_cer': avg_cer,
        'avg_wer': avg_wer,
        'avg_processing_time': avg_processing_time,
        'avg_confidence': avg_confidence,
        'total_texts': total_texts,
        'total_images': len(image_files)
    }

if __name__ == "__main__":
    results = benchmark_ocr() 