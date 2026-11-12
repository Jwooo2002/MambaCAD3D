# ASOCA 중심선 순서 확인 결과

2026-11-07 ImageCAS-X 준비 중 기존 전처리의 물리 범위와 순서를 대조하면서 발견했습니다. 원본 코드와 데이터는 변경하지 않았습니다.

## 직접 확인한 근거

1. 현재 학습 자료 `dataset/v11/0.05_8_32_CTCA_64/Normal/N_10_patch.pkl`의 첫 패치 `[64,32,32]`를 기존 `plane_v5.py` 함수와 `size=8`, `resolution=32` 설정으로 재현했습니다. 전체 패치 MSE와 최대 절대 오차가 모두 **0**이었습니다.
2. 이 패치에 쓰인 `centerlines_corrected_Normal_10_processed_cumulative.pkl` 첫 branch의 첫 중심점 간 거리는 31.66, 16.95, 7.99, 19.36 mm입니다. 혈관을 따라 일정하게 진행하는 이웃 단면 순서와 다릅니다.
3. 같은 종류의 중심선 파일 40명을 검사한 결과, 환자별 연속 중심점 거리의 중앙값은 약 **5.03–22.84 mm**였습니다. 결과는 `external_data/imagecas_x/reports/asoca_centerline_order_audit.csv`에 보존했습니다.
4. 해당 파일을 생성하는 `ASOCADataAccess/sort_cnt_2.py`는 점을 set으로 모으고, set 차집합을 그대로 list로 저장합니다. 집합은 원래 polyline의 혈관 진행 순서를 보존하지 않습니다.

## 리비전에 주는 영향

현재 진행 중인 초기 CV는 기존 입력 자료를 사용한 기준 실행입니다. 이것만으로 원고의 혈관 종축 연속성 주장을 검증했다고 할 수 없습니다. 최종 실험 전에는 VTK/VTP의 line connectivity를 따르는 경로를 사용해 ASOCA와 외부 코호트의 공통 전처리를 확정하고, 수정 전후를 구분해야 합니다.

ImageCAS-X 역시 VTK GetPoints() 배열을 그대로 1D 경로로 사용하면 안 됩니다. 연결된 line/cell의 point IDs를 따라 분지 경로를 구성해야 합니다.

또한 기존 `size=8`은 단면 grid의 **반폭 voxel 수**입니다. 8×8 voxel의 전체 폭을 뜻한다고 자동으로 해석할 수 없습니다. scanner별 실제 spacing을 반영해 mm 단위 촬영 범위를 확인해야 합니다.

이 문서는 현재 자료에서 확인한 전처리 문제의 기록입니다. 최초 투고에 사용한 모든 결과의 정확한 생성 버전·설정을 재확인하고, 원고에서 어떤 결과가 수정되는지 명시해야 합니다.
