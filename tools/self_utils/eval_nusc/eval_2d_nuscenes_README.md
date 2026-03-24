# NuScenes 3D 轉 2D 評估程式說明

## 程式簡介
此程式 (`tools/eval_2d_nuscenes.py`) 的主要功能是將 3D 物件偵測結果（LiDAR 坐標系下的 3D 框）投影到 NuScenes 數據集的 2D 相機影像上，並與 Ground Truth (GT) 3D 框的 2D 投影進行對比，計算 Recall 和 Average IoU。

主要評估類別：
- `Vehicle` (包含 car, truck, bus, trailer, construction_vehicle)
- `Pedestrian` (pedestrian)
- `Cyclist` (bicycle, motorcycle)

## 使用方式
請在終端機執行以下指令：

```bash
python tools/eval_2d_nuscenes.py \
    --pkl <你的預測結果pkl路徑> \
    --nusc_root data/nuscenes/v1.0-mini \
    --version v1.0-mini
```

**參數說明：**
- `--pkl`: 預測結果檔案 (.pkl) 的路徑。
  - 格式需符合 `tools/self_utils/box_det_pkl_read.py` 中的定義。
  - `frame_id` 欄位可以是 NuScenes 的 `sample_token` (32碼 hex) **或者** 對應的 LiDAR 檔名 (例如 `n015-2018...pcd`)。程式會自動進行對應。
- `--nusc_root`: NuScenes 數據集的根目錄。請確認目錄下包含 `maps`, `samples`, `sweeps`, `v1.0-mini` 等資料夾。
- `--version`: 使用的 NuScenes 版本 (例如 `v1.0-mini` 或 `v1.0-trainval`)。

## 參數設定
若需要修改 **置信度 (Confidence)** 或 **TP IoU 閾值**，請直接在程式碼 `tools/eval_2d_nuscenes.py` 中搜尋 `PARAMETER SETTINGS` 區塊進行修改：

```python
    # ==========================================
    #             PARAMETER SETTINGS
    # ==========================================
    # Confidence Thresholds (Default 0.1)
    CONF_THRESHOLDS = {
        'Vehicle': 0.1,
        'Pedestrian': 0.1,
        'Cyclist': 0.1
    }
    
    # IoU Thresholds for True Positive (Default 0.5)
    IOU_THRESHOLDS = {
        'Vehicle': 0.5,
        'Pedestrian': 0.5,
        'Cyclist': 0.5
    }
    
    # Visualization Settings
    VISUALIZE = True       # 設為 True 開啟畫圖功能
    VIS_OUT_DIR = 'vis_results' # 圖片存檔目錄
    VIS_LIMIT = 50         # 限制產生的圖片數量 (避免產生太多)
    # ==========================================
```

## 視覺化結果
若開啟 `VISUALIZE = True`，程式會在 `vis_results/` 目錄下產生圖片：
- **綠色框**: Ground Truth (真值)
- **紅色框**: Prediction (預測結果)

## 函式功能詳解

### 1. `get_2d_iou(box1, box2)`
- **功能**: 計算兩個 2D 邊界框 (Bounding Box) 的 Intersection over Union (IoU)。
- **輸入**: 兩個框的坐標 `[x1, y1, x2, y2]`。
- **輸出**: IoU 值 (0.0 ~ 1.0)。

### 2. `project_box_to_2d(box, intrinsic, imsize)`
- **功能**: 將一個 3D 框投影到 2D 圖像平面上。
- **流程**:
    1. 使用相機內參 (Intrinsic Matrix) 將 3D 角點轉換為 2D 圖像坐標。
    2. 將超出圖像範圍的框進行裁剪 (Clip)，限制在 `[0, 0, width, height]` 範圍內。
    3. 取投影點的最小/最大 x, y 值，形成外接矩形 `[x1, y1, x2, y2]`。
    4. 若投影後的框面積過小則過濾掉。
- **輸入**: `box` (NuScenes Box 物件), `intrinsic` (內參矩陣), `imsize` (圖像尺寸)。
- **輸出**: 2D 框 `[x1, y1, x2, y2]` 或 `None`。

### 3. `get_sample_data(nusc, sample_token, box_lidar_frame)`
- **功能**: 將單個 LiDAR 坐標系下的預測 3D 框，轉換並投影到該幀 (Sample) 對應的所有相機 (CAM_FRONT, CAM_BACK 等) 的 2D 圖像上。
- **流程**:
    1. 獲取該 Sample 的 LiDAR 和各個相機的校正參數 (Calibrated Sensor) 和自身姿態 (Ego Pose)。
    2. 執行坐標轉換：
       `Lidar 坐標` -> `Lidar Ego` -> `Global` -> `Cam Ego` -> `Cam 坐標` -> `Image 2D`.
    3. 調用 `project_box_to_2d` 獲取最終 2D 框。
- **輸入**: `nusc` (數據集實例), `sample_token` (幀ID), `box_lidar_frame` (LiDAR 坐標系下的 Box)。
- **輸出**: 字典 `{ 'CAM_FRONT': [x1,y1,x2,y2], ... }`。

### 4. `eval_2d(pkl_file, nusc, score_thresh=0.1)`
- **功能**: 這是程式的主流程函式。
- **流程**:
    1. 載入預測的 `.pkl` 檔案。
    2. 遍歷每一個預測幀 (Frame)：
        - **處理 GT**: 讀取該幀的 GT 3D 標註，將其投影到所有相機圖像上，作為真值 (Ground Truth)。
        - **處理預測**: 讀取預測的 3D 框，利用 `get_sample_data` 將其投影到所有相機圖像上。
    3. **進行匹配 (Matching)**:
        - 在每個相機圖像上，將預測框與 GT 框進行 IoU 比對。
        - 若 IoU > 0.5 視為 True Positive (TP)。
        - 統計 TP (正確偵測)、DT (預測總數)、GT (真值總數)。
    4. 計算並列印最終表格 (Recall, Avg IoU)。

### 5. `parse_args()`
- **功能**: 處理命令行輸入的參數 (pkl 路徑, 數據集路徑等)。

## 輸出範例
程式執行完畢後會顯示如下表格：
```text
Class                     | Total TP   | Total DT   | Total GT   | Recall     | Avg IoU (TPs)  
-----------------------------------------------------------------------------------------------
Vehicle                   | 501        | 1229       | 2174       | 0.2305     | 0.7233         
Pedestrian                | 96         | 138        | 1595       | 0.0602     | 0.6004         
Cyclist                   | 0          | 0          | 193        | 0.0000     | 0.0000         
```
