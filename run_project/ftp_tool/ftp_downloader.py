import os
from ftplib import FTP

# --- 設定 ---
FTP_HOST = "140.124.182.50"
FTP_USER = "113C52042"
FTP_PASS = "Jason#910714"
#REMOTE_BASE_PATH = "/1323LAB_FTP/Labeled_RealCar_Dataset/"  # 你要掃描的遠端根目錄
REMOTE_BASE_PATH = "/OpenDataSet/義隆實車資料集/data/2025-09-25"
LOCAL_BASE_PATH = r"D:\Dataset\real\2025-09-25"   # 下載到哪裡
TARGET_DIRS = {"imu", "VLS128_pcdnpy", "image"}     # 你要找的資料夾名字
ALLOWED_EXTENSIONS = {".pcd", ".txt", ".jpg", ".png"} # 只下載這些副檔名的檔案


def download_dir(ftp, remote_path, local_path):
    """
    一個純粹的遞迴下載函式。
    現在它只下載 ALLOWED_EXTENSIONS 中指定的檔案類型。
    """
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        # 註解：我們只在真的要下載東西時才印出建立目錄的訊息，
        # 但為了確保路徑存在，還是要先建立。

    ftp.cwd(remote_path)
    items = ftp.nlst()

    for item in items:
        local_item_path = os.path.join(local_path, item)
        remote_item_path = f"{remote_path}/{item}"

        try:
            ftp.cwd(remote_item_path)
            # 如果是目錄，永遠遞迴進去，因為目標檔案可能在子目錄裡
            download_dir(ftp, remote_item_path, local_item_path)
            ftp.cwd("..")
        except Exception:
            # 如果是檔案，檢查副檔名
            if any(item.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                print(f"Downloading file: {remote_item_path} to {local_item_path}")
                with open(local_item_path, 'wb') as f:
                    ftp.retrbinary(f"RETR {item}", f.write)

def find_and_mirror_dirs(ftp, remote_path, local_path):
    """
    遞迴尋找目標資料夾。
    如果找到了，就呼叫 download_dir 來處理。
    """
    original_dir = ftp.pwd()
    try:
        ftp.cwd(remote_path)
    except Exception as e:
        print(f"Cannot access remote directory: {remote_path}. Skipping. Error: {e}")
        return

    items = ftp.nlst()
    
    for item in items:
        # 忽略 '.' 和 '..'
        if item in ['.', '..']:
            continue

        remote_item_path = f"{remote_path.rstrip('/')}/{item}"
        
        # 檢查這是不是一個目錄
        is_directory = False
        try:
            ftp.cwd(remote_item_path)
            is_directory = True
            ftp.cwd("..") # 檢查完就回來
        except Exception:
            is_directory = False
            
        if is_directory:
            if item in TARGET_DIRS:
                print(f"Found target directory: {remote_item_path}")
                # 本地路徑應該包含其父目錄結構
                local_target_path = os.path.join(local_path, os.path.relpath(remote_item_path, REMOTE_BASE_PATH))
                download_dir(ftp, remote_item_path, local_target_path)
            else:
                # 沒找到，就繼續往深處找
                find_and_mirror_dirs(ftp, remote_item_path, local_path)

    # 恢復到進入此函式前的目錄，確保遞迴的穩定性
    ftp.cwd(original_dir)


def main():
    """主函式，處理連線和啟動邏輯。"""
    try:
        with FTP(FTP_HOST) as ftp:
            ftp.login(user=FTP_USER, passwd=FTP_PASS)
            print(f"Connected to {FTP_HOST}")
            
            # 確保本地根目錄存在
            if not os.path.exists(LOCAL_BASE_PATH):
                os.makedirs(LOCAL_BASE_PATH)
            
            print(f"Starting scan in {REMOTE_BASE_PATH}...")
            find_and_mirror_dirs(ftp, REMOTE_BASE_PATH, LOCAL_BASE_PATH)
            print("Done.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()