# ImageCAS-X 외부 검증 데이터 준비

시작일: 2026-11-07. 저장 위치는 `external_data/imagecas_x/`이며, 대용량 데이터는 소스 관리에서 제외합니다.

**완료 상태:** 원본 CCTA 1,000건 다운로드·추출, 공식 제외 200명을 뺀 외부 코호트 800명 연결, 800명 전체의 좌표 검사와 중심선 그래프 생성·읽기 검증을 마쳤습니다. 정상 412명·질환 388명입니다. ZIP 목록의 파일 크기와 모두 일치하며, 동일 CRC/크기의 중복 파일 후보는 없었습니다. 최종 검증 기록은 `reports/final_readiness_check.json`입니다. 모델용 패치 변환은 아래 남은 작업에 해당합니다.

데이터 디스크가 SMR HDD여서 대량 파일 쓰기와 압축 해제에 병목이 발생했습니다. 새로 받는 원본 분할 파일과 추출 영상은 영구 SSD 폴더 `/home/jiwoong/Datasets/ImageCAS-X`에 저장하고 작업 폴더에서 symlink로 연결합니다. 이미 검증한 초기 HDD 분할 파일은 `downloads/imagecas_v3_hdd_initial/`에 보존해 연결합니다. 상세 배치는 `provenance/storage_layout.json`에 기록합니다.

## 고정한 배포 버전

- ImageCAS-X **v3**, 2026-11-02: <https://zenodo.org/records/23894668>.
- 주석: `ImageCAS-X_dataset_v3.zip`, 934,085,540 bytes.
- 공개 MD5: `4e7ec922065905fe29cc820276e6f283`.
- 원본 CCTA: ImageCAS Kaggle dataset **version 3**.
- <https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas>
- 원본 영상 압축파일은 5개 묶음, 각각 5분할로 총 25개 / 89,085,661,377 bytes입니다.
- 혈관 주석에는 수정된 ImageCAS-X의 내강/분지 라벨을 사용합니다.
- 공개 segmentation 모델 가중치는 이번 데이터 준비에 필요하지 않아 다운로드에 포함하지 않습니다.

## 환자 구성

원본 영상은 1,000명입니다. 저자의 `exclude.txt`에 있는 200명을 제외하면 **정상 412명·질환 388명, 총 800명**입니다. 제외 200명의 비어 있는 Disease 값을 정상으로 채우지 않습니다.

원래 segmentation benchmark의 train/val/test는 560/80/160명입니다. 이 구분은 출처 정보로 CSV에 보존합니다. 이번 리비전에서는 800명 모두 external test로 지정하며, 해당 라벨로 모델·판정 기준·aggregation을 최적화하지 않습니다.

`cohort_membership.csv`의 SHA-256을 최초 생성 시 고정합니다. 평가 결과를 확인한 뒤 대상 환자를 임의로 바꾸지 않기 위한 기록입니다.

## 준비 과정

1. 공식 배포처에서 HTTP range를 사용해 이어받을 수 있도록 다운로드합니다.
2. 주석 압축파일의 공개 MD5와 ZIP CRC를 확인합니다.
3. Kaggle이 각 대용량 파일에 씌우는 ZIP 포장을 풀고, 원본 파일의 크기·CRC를 확인합니다. 원본 25개 분할 파일 각각의 SHA-256도 저장합니다.
4. SSD 임시 폴더 `/tmp/vmamba_imagecas_x_transport`에서 병렬 다운로드를 받고, 영구 SSD 폴더에 검증된 원본 분할 파일을 저장합니다. 다운로드 저장이 끝나면 공식 Linux 7-Zip으로 CCTA `*.img.nii.gz`만 추출합니다. 원본 분할 파일은 보존하고, 중복된 Kaggle ZIP 포장만 제거합니다.
5. CT·혈관 마스크·좌우 중심선·환자 라벨을 scan ID로 연결하고, 포함 800명과 제외 200명의 목록을 만듭니다.
6. CT/마스크의 크기·spacing·origin·direction을 대조합니다. 중심선이 LPS인지 RAS→LPS 변환이 필요한지 혈관 내강과의 겹침으로 확인합니다. 의심 사례는 기록하며, 환자를 추가로 자동 제외하지 않습니다.
7. 환자별 중심선 그래프를 LPS mm 좌표와 실제 line 연결 관계로 내보냅니다. 원래 방향을 가진 edge와 중복 제거한 무방향 edge를 함께 보존하고, node degree는 중복 제거한 연결로 계산합니다.

## 실행과 진행 확인

기존 `journal_preprocess` 환경을 사용하며, 학습 중인 VMamba 환경은 변경하지 않습니다.

```bash
cd /mnt/sda2/Jiwoong/Results/VMamba/kfold_simplescan_revision
/home/jiwoong/anaconda3/envs/journal_preprocess/bin/python -u scripts/fetch_imagecas_x.py --workers 8
python3 monitor_data.py
```

다운로드와 압축 해제가 끝나면 전체 800명의 좌표 검증이 자동으로 실행됩니다. 다시 실행할 때는 먼저 기존 다운로드 프로세스를 확인해야 합니다. 같은 파일을 여러 프로세스에서 동시에 받지 않습니다.

좌표 검증만 다시 실행:

```bash
/home/jiwoong/anaconda3/envs/journal_preprocess/bin/python scripts/prepare_imagecas_x.py --validate-geometry --workers 4
```

다운로드 중 소수 사례 확인:

```bash
/home/jiwoong/anaconda3/envs/journal_preprocess/bin/python scripts/prepare_imagecas_x.py --validate-geometry --available-only --limit 4 --workers 2
```

## 결과 위치

- `downloads/`: 배포 압축파일과 검증된 원본 분할 파일.
- `provenance/`: 공식 배포 정보, 압축파일 목록, 버전 정보.
- `raw/annotations/ImageCAS-X_dataset_v3/`: 혈관 마스크, 중심선, 표면, 환자 정보와 원래 split.
- `raw/imagecas_volumes/`: 원본 CCTA 영상.
- `prepared/external_cohort.csv`: 외부 평가 800명의 연결 목록.
- `prepared/cohort_membership.csv`: 고정한 환자 ID·라벨·평가 역할.
- `prepared/excluded_scans.csv`: 저자가 제외한 200명과 사유.
- `prepared/volumes/`: 대상 800명의 영상에 대한 symlink. 영상을 중복 복사하지 않습니다.
- `prepared/graphs/<scan_id>.npz`: 좌우 중심선의 LPS mm 좌표, 연결 edge, node degree, 원래 point 속성, CT의 spacing/origin/direction. `numpy.load(..., allow_pickle=False)`로 읽습니다.
- `reports/download.log`, `download_state/`: 다운로드 진행과 이어받기 정보.
- `reports/geometry_qc.csv`, `geometry_qc_summary.json`: 전체 환자의 물리 좌표 검증 결과.
- `reports/image_integrity_inventory.csv`, `image_integrity_summary.json`: 원본 ZIP 목록과 추출 크기 대조, 동일 파일 후보 검사 결과.
- `prepared/external_cohort_manifest.json`: 파일 준비, 환자 구성 hash, 검증 상태.

## 사용 범위와 남은 작업

Disease는 **환자 단위** 라벨입니다. 혈관 내강·분지 마스크를 plaque/stenosis 위치 정답으로 해석하면 안 됩니다. 배포된 reference 중심선을 사용하므로 완전 자동 중심선 추출 성능을 검증한 것도 아닙니다.

이번 준비 단계는 원본 CCTA와 대응 주석의 정리·좌표 검증입니다. 모델 입력인 `64×32×32` 패치 PKL로 변환하기 전에는 ASOCA의 실제 물리적 촬영 범위와 중심선 sampling을 확정해야 합니다. 다른 scanner에서 voxel 개수만 맞추면 물리적 범위가 달라질 수 있습니다.
