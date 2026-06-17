"""노브 위젯 모음 (슬라이더/토글 패널).

각 노브가 pipeline 모듈 하나에 대응:
  confidence/min_size/brightness -> filters
  debounce N프레임             -> debounce
  frame_skip/resize            -> infer 입력 단 (속도 레버)
  ROI 그리기                    -> roi
  counting/dwell 표시           -> tracking
  시각화 토글(박스/마스크/ID/conf)
"""
