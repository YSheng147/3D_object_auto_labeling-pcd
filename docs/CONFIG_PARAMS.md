# 設定檔參數說明

## 1. `tools/cfgs/dataset_configs/custom_dataset_da.yaml`

資料集載入與前處理設定，供推論與訓練共用。

### 基本設定

| 參數 | 說明 |
|------|------|
| `DATASET` | 使用的 Dataset class，固定為 `CustomDataset` |
| `DATA_PATH` | 資料集根目錄（`pointcloud/` 層），由 `run_MS3D_v2.py` 自動修改 |
| `SHIFT_COOR` | `[x, y, z]` 坐標平移。**z 值必須填入 LiDAR 離地高度（單位：公尺）**，將點雲平移使地面對齊 z=0，與預訓練模型的訓練假設一致。例：`[0, 0, 1.8]` 表示 LiDAR 距地 1.8 m |
| `POINT_CLOUD_RANGE` | `[xmin, ymin, zmin, xmax, ymax, zmax]` 有效偵測範圍（公尺）。提供三種版本：以車輛為中心、向右偏、向前偏，依場景需求切換 |
| `MAX_SWEEPS` | 積累幾幀點雲（multi-sweep），增加遠距離稀疏點的密度。值越大點越密，但計算量也越大 |

### 資料分割

| 參數 | 說明 |
|------|------|
| `DATA_SPLIT` | 指定 train/test 各使用哪個分割集（`train` / `val`） |
| `SAMPLED_INTERVAL` | 取樣間隔，`1` = 每幀都用，`2` = 每隔一幀取一幀 |
| `INFO_PATH` | train/test 對應的 dataset info `.pkl` 檔路徑（由 `create_infos` 生成） |

### 特徵編碼

| 參數 | 說明 |
|------|------|
| `POINT_FEATURE_ENCODING.encoding_type` | 點雲編碼方式，`absolute_coordinates_encoding` = 使用絕對坐標 |
| `POINT_FEATURE_ENCODING.used_feature_list` | 實際輸入模型的特徵：`[x, y, z, timestamp]` |
| `POINT_FEATURE_ENCODING.src_feature_list` | 來源 `.pcd` 檔中的特徵欄位名稱，需與 `used_feature_list` 對應 |

### 類別設定

| 參數 | 說明 |
|------|------|
| `CLASS_NAMES` | 輸出偵測的目標類別：`['Vehicle', 'Pedestrian', 'Cyclist']` |
| `CLASS_MAPPING` | 將原始標籤（car、truck、bus、bicycle 等）統一對應到上述三類 |

### 資料增強（`DATA_AUGMENTOR`）

僅訓練時有效，推論時不套用。

| 增強方式 | 參數 | 說明 |
|----------|------|------|
| `random_world_flip` | `ALONG_AXIS_LIST: ['x', 'y']` | 沿 x 或 y 軸隨機翻轉整個點雲場景 |
| `random_world_rotation` | `WORLD_ROT_ANGLE: [-0.785, 0.785]` | 隨機旋轉 ±45°（±π/4 弧度） |
| `random_world_scaling` | `WORLD_SCALE_RANGE: [0.95, 1.05]` | 隨機縮放至 95%–105% |

### 資料處理（`DATA_PROCESSOR`）

| 處理步驟 | 參數 | 說明 |
|----------|------|------|
| `mask_points_and_boxes_outside_range` | `REMOVE_OUTSIDE_BOXES: true` | 移除超出 `POINT_CLOUD_RANGE` 的點與框 |
| `shuffle_points` | `SHUFFLE_ENABLED` | 訓練時隨機打亂點的順序，推論時不打亂 |
| `transform_points_to_voxels` | `VOXEL_SIZE: [0.1, 0.1, 0.15]` | 體素尺寸（x, y, z 方向，單位：公尺） |
| | `MAX_POINTS_PER_VOXEL: 5` | 每個體素最多保留 5 個點 |
| | `MAX_NUMBER_OF_VOXELS` | 訓練 80,000 個體素，推論 90,000 個體素 |

---

## 2. `tools/cfgs/target_custom/label_generation/round1/cfgs/ps_config.yaml`

MS3D++ 偽標籤生成流程的主要參數設定，控制三個階段：KBF 融合、追蹤、時序精煉。

### 路徑設定

| 參數 | 說明 |
|------|------|
| `DETS_TXT` | `ensemble_detections.txt` 的路徑，內含所有模型 `result.pkl` 的絕對路徑清單 |
| `SAVE_DIR` | 所有中間結果與最終偽標籤的輸出目錄 |
| `DATA_CONFIG_PATH` | 指向 `custom_dataset_da.yaml`（相對於 `tools/` 目錄） |

### 偽標籤信心分數閾值（`PS_SCORE_TH`）

順序為 `[Vehicle, Pedestrian, Cyclist]`。

| 參數 | 說明 |
|------|------|
| `POS_TH` | **正樣本閾值**，高於此值的框視為高品質偽標籤保留。建議從嚴（如 0.7/0.6/0.5），false positive 一旦進入後續訓練很難消除 |
| `NEG_TH` | **負樣本閾值**，低於此值的框直接丟棄視為噪音。介於 NEG_TH 與 POS_TH 之間的框為不確定區間 |

### KBF 集成融合（`ENSEMBLE_KBF`）

| 參數 | 說明 |
|------|------|
| `DISCARD` | `[veh, ped, cyc]` **投票數下限**：少於此數量的模型同意時丟棄該框（類似 NMS 前的最低共識要求） |
| `RADIUS` | `[veh, ped, cyc]` KBF 核寬度（公尺），控制多個重疊框合併的距離容忍度。車輛較大（1.5 m），行人較小（0.3 m） |
| `NMS` | `[veh, ped, cyc]` 合併後 NMS 的 IoU 閾值，消除剩餘重疊框 |

### 追蹤器設定（`TRACKING`）

分四個追蹤器：`VEH_ALL`（所有車輛）、`VEH_STATIC`（靜態車輛）、`PEDESTRIAN`、`CYCLIST`。

每個追蹤器有兩組子設定：

#### `RUNNING`（主追蹤器）

| 參數 | 說明 |
|------|------|
| `SCORE_TH` | 啟動一條新軌跡所需的最低偵測分數 |
| `MAX_AGE_SINCE_UPDATE` | 軌跡最多幾幀沒有偵測更新就刪除（容許短暫遮擋） |
| `MIN_HITS_TO_BIRTH` | 需連續命中幾次才正式建立軌跡（避免偶發誤報） |
| `ASSO` | 偵測框與軌跡的關聯方法：`giou`（廣義 IoU，適合動態目標）、`iou_2d`（2D IoU，適合靜態車輛） |
| `ASSO_TH` | 關聯距離閾值，超過此值不允許關聯 |

#### `REDUNDANCY`（冗餘追蹤器）

補充分數較低但可能真實存在的目標，尤其用於填補主追蹤器遺漏的幀。

| 參數 | 說明 |
|------|------|
| `SCORE_TH` | 冗餘追蹤器的啟動分數（通常低於 `RUNNING.SCORE_TH`） |
| `MAX_REDUNDANCY_AGE` | 冗餘軌跡的最大存活幀數 |
| `ASSO_TH` | 冗餘關聯閾值 |

### 時序精煉（`TEMPORAL_REFINEMENT`）

#### `TRACK_FILTERING`（軌跡過濾）

| 參數 | 說明 |
|------|------|
| `MIN_NUM_STATIC_VEH_TRACKS` | 靜態車輛軌跡至少需出現幾幀才保留（10） |
| `MIN_NUM_PED_TRACKS` | 行人軌跡最少出現幀數（10） |
| `MIN_NUM_CYC_TRACKS` | 騎士軌跡最少出現幀數（5，因騎士較稀少） |
| `MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_ALL` | 軌跡中需有幾幀分數超過 `POS_TH` 才保留（車輛：4 幀） |
| `MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_VEH_STATIC` | 靜態車輛需有幾幀超過 POS_TH（5 幀，要求更嚴） |
| `MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_PED` | 行人需有幾幀超過 POS_TH（2 幀） |
| `MIN_DETS_ABOVE_POS_TH_FOR_TRACKS_CYC` | 騎士需有幾幀超過 POS_TH（5 幀） |
| `USE_STATIC_PED_TRACKS` | 是否使用靜止行人的軌跡（`false` = 不使用，避免把固定假目標放進偽標籤） |

#### `ROLLING_KBF`（滑動視窗 KBF）

在軌跡的時序維度上再做一次 KBF 精煉，融合同一物體在不同幀的框。

| 參數 | 說明 |
|------|------|
| `MIN_STATIC_SCORE` | 靜態物體被納入 ROLLING_KBF 的最低分數（0.8） |
| `ROLLING_KDE_WINDOW` | 滑動視窗大小（幀數），窗內的框一起做 KDE 融合（15 幀）。越大越穩定但越難跟上形狀變化 |

#### `PROPAGATE_BOXES`（框的時序傳播）

對靜態目標，將已偵測到的框向前後幀傳播，補全短暫未偵測到的幀。

| 參數 | 說明 |
|------|------|
| `MIN_STATIC_TRACKS` | 觸發傳播機制所需的最少靜態軌跡數（10），太少靜態目標時不做傳播 |
| `N_EXTRA_FRAMES` | 向前後各傳播幾幀（10 幀） |
| `DEGRADE_FACTOR` | 每傳播一幀分數乘以此衰減係數（0.99），距離越遠的幀分數越低 |
| `MIN_SCORE_CLIP` | 傳播分數的下限（0.3），避免分數衰減至 0 |
