import argparse
import pickle
import numpy as np
import os
import cv2
from collections import defaultdict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion
from shapely.geometry import Polygon

def get_2d_iou(box1, box2):
    """
    Calculate IoU between two 2D boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def project_box_to_2d(box, intrinsic, imsize):
    """
    Project a 3D box to a 2D bounding box [x1, y1, x2, y2] in image coordinates.
    """
    # Create a new box to avoid modifying the original one
    box = box.copy()
    
    # Check if box is visible
    # if not box_in_image(box, intrinsic, imsize, vis_level=BoxVisibility.ANY):
    #     return None

    corners_3d = box.corners()
    corners_2d = view_points(corners_3d, intrinsic, normalize=True)[:2, :]

    # Clip to image boundaries
    corners_2d[0, :] = np.clip(corners_2d[0, :], 0, imsize[0])
    corners_2d[1, :] = np.clip(corners_2d[1, :], 0, imsize[1])

    x1 = np.min(corners_2d[0, :])
    y1 = np.min(corners_2d[1, :])
    x2 = np.max(corners_2d[0, :])
    y2 = np.max(corners_2d[1, :])

    # Filter out invalid boxes (area 0)
    if (x2 - x1) <= 1 or (y2 - y1) <= 1:
        return None

    return [x1, y1, x2, y2]

def get_sample_data(nusc, sample_token, box_lidar_frame):
    """
    Map Lidar box to all cameras and project to 2D.
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    sd_lidar = nusc.get('sample_data', lidar_token)
    cs_lidar = nusc.get('calibrated_sensor', sd_lidar['calibrated_sensor_token'])
    pose_lidar = nusc.get('ego_pose', sd_lidar['ego_pose_token'])

    camera_results = {}
    
    cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
               'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for cam in cameras:
        if cam not in sample['data']:
            continue
        
        cam_token = sample['data'][cam]
        sd_cam = nusc.get('sample_data', cam_token)
        cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
        pose_cam = nusc.get('ego_pose', sd_cam['ego_pose_token'])
        imsize = (sd_cam['width'], sd_cam['height'])
        
        # Transform box from Lidar Frame -> Det Ego Frame -> Global -> Cam Ego -> Cam
        # But wait, input boxes are in LIDAR frame of the sample?
        # Typically detection results in simple pkl are often in Lidar Frame.
        
        # Lidar -> Ego (Lidar Time)
        box_ego = box_lidar_frame.copy()
        box_ego.rotate(Quaternion(cs_lidar['rotation']))
        box_ego.translate(np.array(cs_lidar['translation']))
        
        # Ego (Lidar Time) -> Global
        box_global = box_ego.copy()
        box_global.rotate(Quaternion(pose_lidar['rotation']))
        box_global.translate(np.array(pose_lidar['translation']))
        
        # Global -> Ego (Cam Time)
        box_cam_ego = box_global.copy()
        box_cam_ego.translate(-np.array(pose_cam['translation']))
        box_cam_ego.rotate(Quaternion(pose_cam['rotation']).inverse)
        
        # Ego (Cam Time) -> Cam
        box_cam = box_cam_ego.copy()
        box_cam.translate(-np.array(cs_cam['translation']))
        box_cam.rotate(Quaternion(cs_cam['rotation']).inverse)
        
        # Cam -> Image
        box_2d = project_box_to_2d(box_cam, np.array(cs_cam['camera_intrinsic']), imsize)
        
        if box_2d is not None:
            camera_results[cam] = box_2d
            
    return camera_results

def eval_2d(pkl_file, nusc):
    with open(pkl_file, 'rb') as f:
        predictions = pickle.load(f)

    # Stats
    metrics = defaultdict(lambda: {'TP': 0, 'DT': 0, 'GT': 0, 'IoU_Sum': 0})
    
    # ==========================================
    #             PARAMETER SETTINGS
    # ==========================================
    # Confidence Thresholds (Default 0.1)
    CONF_THRESHOLDS = {
        'Vehicle': 0.5,
        'Pedestrian': 0.3,
        'Cyclist': 0.3
    }
    
    # IoU Thresholds for True Positive (Default 0.5)
    IOU_THRESHOLDS = {
        'Vehicle': 0.5,
        'Pedestrian': 0.3,
        'Cyclist': 0.3
    }
    
    # Visualization Settings
    VISUALIZE = True       # Set to True to enable saving images with boxes
    VIS_OUT_DIR = 'vis_results' # Directory to save visualization results
    VIS_LIMIT = 500         # Limit number of saved images to avoid flooding
    # ==========================================
    
    if VISUALIZE:
        if not os.path.exists(VIS_OUT_DIR):
            os.makedirs(VIS_OUT_DIR)
        print(f"Visualization enabled. Saving to {VIS_OUT_DIR}...")

    print(f"Confidence Thresholds: {CONF_THRESHOLDS}")
    print(f"IoU Thresholds: {IOU_THRESHOLDS}")
    
    # Class mapping from nuscenes_dataset_da.yaml / User request
    # Pred classes: Vehicle, Pedestrian, Cyclist
    # GT classes in NuScenes are granular (car, truck, ...). We need to map them.
    class_map = {
        'car': 'Vehicle', 'truck': 'Vehicle', 'bus': 'Vehicle', 'trailer': 'Vehicle', 'construction_vehicle': 'Vehicle',
        'pedestrian': 'Pedestrian',
        'bicycle': 'Cyclist', 'motorcycle': 'Cyclist'
    }
    
    valid_pred_classes = ['Vehicle', 'Pedestrian', 'Cyclist']

    print(f"Evaluating {len(predictions)} frames...")
    
    # Create a mapping from filename to sample token
    print("Creating filename to sample token mapping...")
    filename_to_token = {}
    for sample in nusc.sample:
        lidar_token = sample['data']['LIDAR_TOP']
        sd_lidar = nusc.get('sample_data', lidar_token)
        # filename is relative path, e.g. "samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin"
        # The pkl has "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd"
        # We need to match the filename part.
        base_filename = os.path.basename(sd_lidar['filename'])
        # The pkl filename might not have extension or might have .pcd
        # Let's clean both side to match
        # pkl: n015...pcd
        # nusc: n015...pcd.bin
        
        # Key: filename witout extension (or just the main part)
        # Actually safer to look for the timestamp in filename or just direct match if pkl filename is unique enough. 
        # based on user pkl: n015...pcd. 
        # nusc filename: samples/LIDAR_TOP/n015...pcd.bin
        
        # Let's try to match by removing .bin from nusc filename and comparing with pkl filename
        
        nusc_fname = os.path.basename(sd_lidar['filename'])
        if nusc_fname.endswith('.bin'):
            nusc_fname = nusc_fname[:-4] # remove .bin
            
        filename_to_token[nusc_fname] = sample['token']

    print(f"Mapped {len(filename_to_token)} sample tokens.")
    
    vis_count = 0

    for frame in predictions:
        try:
            frame_id = frame['frame_id']
            # Try to find token
            if frame_id in filename_to_token:
                sample_token = filename_to_token[frame_id]
            else:
                # Try simple token verify
                try:
                    nusc.get('sample', frame_id)
                    sample_token = frame_id
                except:
                    # Try fuzzy match?
                    # maybe frame_id has .pcd and map has without?
                    # or frame_id is just stem?
                    if frame_id.endswith('.pcd'):
                         fid = frame_id
                    else:
                         fid = frame_id + '.pcd'
                    
                    if fid in filename_to_token:
                        sample_token = filename_to_token[fid]
                    else:
                        # Skip if not found
                        # print(f"Warning: frame_id {frame_id} not found in NuScenes {nusc.version}")
                        continue

            # Verify token exists
            try:
                nusc.get('sample', sample_token)
            except:
                continue

            # Process Ground Truth for this sample
            # Get all GT boxes in global frame, then project to all cameras
            sample = nusc.get('sample', sample_token)
            
            gt_boxes_2d_by_cam = defaultdict(list) # cam -> list of {'box': [x1,y1,x2,y2], 'name': Class}
            
            # Get annotations
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cat_name = ann['category_name'].split('.')[0] # e.g. vehicle.car -> vehicle
                
                # Map detailed class to target class
                target_class = None
                for k, v in class_map.items():
                   if k in ann['category_name']:
                       target_class = v
                       break
                
                if target_class is None: 
                    continue

                # Project GT to all cameras
                # GT boxes are Global.
                box_gt_global = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
                
                cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
                
                for cam in cameras:
                    if cam not in sample['data']: continue
                    cam_token = sample['data'][cam]
                    sd_cam = nusc.get('sample_data', cam_token)
                    cs_cam = nusc.get('calibrated_sensor', sd_cam['calibrated_sensor_token'])
                    pose_cam = nusc.get('ego_pose', sd_cam['ego_pose_token'])
                    imsize = (sd_cam['width'], sd_cam['height'])
                    
                    # Global -> Ego (Cam Time)
                    box_cam_ego = box_gt_global.copy()
                    box_cam_ego.translate(-np.array(pose_cam['translation']))
                    box_cam_ego.rotate(Quaternion(pose_cam['rotation']).inverse)
                    
                    # Ego (Cam Time) -> Cam
                    box_cam = box_cam_ego.copy()
                    box_cam.translate(-np.array(cs_cam['translation']))
                    box_cam.rotate(Quaternion(cs_cam['rotation']).inverse)
                    
                    # Cam -> Image
                    # Check if center is in front of camera
                    if box_cam.center[2] <= 0: continue

                    box_2d = project_box_to_2d(box_cam, np.array(cs_cam['camera_intrinsic']), imsize)
                    
                    if box_2d:
                        gt_boxes_2d_by_cam[cam].append({'box': box_2d, 'name': target_class})
                        metrics[target_class]['GT'] += 1

            # Process Predictions
            # Preds are in array inputs: boxes_lidar (N, 7), name (N), score (N)
            pred_boxes = frame['boxes_lidar']
            pred_names = frame['name']
            pred_scores = frame['score']

            # Pred cache by cam
            pred_boxes_2d_by_cam = defaultdict(list)

            for i in range(len(pred_names)):
                cls_name = pred_names[i]
                if cls_name not in valid_pred_classes:
                    continue
                
                # Check Class-specific Confidence Threshold
                conf_th = CONF_THRESHOLDS.get(cls_name, 0.1)
                if pred_scores[i] < conf_th:
                    continue
                
                # Input format from box_det_pkl_read.py: [x, y, z, dx, dy, dz, heading]
                # OpenPCDet NuScenes models typically output in a shifted frame where Z += 1.8.
                # nuscenes_dataset_da.yaml: SHIFT_COOR: [0.0, 0.0, 1.8]
                # We must unshift Z to get back to Lidar Frame (which is ~1.84m above ground).
                
                loc = pred_boxes[i][:3].copy()
                loc[2] -= 1.8 # Unshift Z
                
                size = [pred_boxes[i][4], pred_boxes[i][3], pred_boxes[i][5]] # dy, dx, dz
                rot = Quaternion(axis=[0, 0, 1], radians=pred_boxes[i][6])
                
                box_lidar = Box(loc, size, rot)

                # Project to all cameras
                # We need to know which camera this box projects to.
                # get_sample_data returns dict {cam: box_2d}
                
                # Careful: The helper get_sample_data handles coordinate transforms.
                projected_results = get_sample_data(nusc, sample_token, box_lidar)
                
                for cam, box_2d in projected_results.items():
                    pred_boxes_2d_by_cam[cam].append({'box': box_2d, 'name': cls_name, 'matched': False})
                    metrics[cls_name]['DT'] += 1

            # Calculate IoU and Matches Per Camera
            for cam in gt_boxes_2d_by_cam.keys():
                gts = gt_boxes_2d_by_cam[cam]
                preds = pred_boxes_2d_by_cam.get(cam, [])
                
                # Sort preds by score? We don't have score in this separate list, assume all passed threshold.
                # Ideally we match high score first, but simple greedy is ok.
                
                # For each GT, find best matching Pred
                processed_preds = [False] * len(preds)
                
                for gt in gts:
                    gt_box = gt['box']
                    gt_cls = gt['name']
                    
                    best_iou = 0
                    best_idx = -1
                    
                    for idx, pred in enumerate(preds):
                        if processed_preds[idx]: continue
                        if pred['name'] != gt_cls: continue
                        
                        iou = get_2d_iou(gt_box, pred['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                    
                    # Check Class-specific IoU Threshold
                    iou_th = IOU_THRESHOLDS.get(gt_cls, 0.5)
                    if best_iou >= iou_th:
                        metrics[gt_cls]['TP'] += 1
                        metrics[gt_cls]['IoU_Sum'] += best_iou
                        processed_preds[best_idx] = True
                    else:
                        pass # FN is implicitly Total GT - TP

            # Visualization Logic
            if VISUALIZE and vis_count < VIS_LIMIT:
                # We need to render images for this sample if there are detections or GT
                # To save time, only render if we have something to draw
                
                # Check if we have any preds or GT
                has_content = False
                for cam in gt_boxes_2d_by_cam:
                    if len(gt_boxes_2d_by_cam[cam]) > 0: has_content = True
                for cam in pred_boxes_2d_by_cam:
                    if len(pred_boxes_2d_by_cam[cam]) > 0: has_content = True
                
                if has_content:
                    vis_count += 1
                    
                    # Iterate cameras
                    # Need to get camera tokens again or cache them?
                    # We can get them from the sample record
                    sample = nusc.get('sample', sample_token)
                    
                    for cam_name, cam_token in sample['data'].items():
                        if 'CAM' not in cam_name: continue
                        
                        gts = gt_boxes_2d_by_cam.get(cam_name, [])
                        preds = pred_boxes_2d_by_cam.get(cam_name, [])
                        
                        if len(gts) == 0 and len(preds) == 0:
                            continue
                            
                        # Load Image
                        sd_cam = nusc.get('sample_data', cam_token)
                        filename = sd_cam['filename']
                        img_path = os.path.join(nusc.dataroot, filename)
                        
                        if not os.path.exists(img_path):
                            print(f"Warning: Image not found {img_path}")
                            continue
                            
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                            
                        # Draw GT (Green)
                        for gt in gts:
                            box = gt['box'] # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, gt['name'], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Draw Pred (Blue for TP?, Red for FP?)
                        # We don't track TP/FP per box easily here unless we redo matching or store it.
                        # For now, just draw all Preds in Red (or maybe distinct color)
                        # Let's say Red
                        for pred in preds:
                            box = pred['box']
                            x1, y1, x2, y2 = map(int, box)
                            # label = f"{pred['name']} {pred.get('score', 0):.2f}"
                            label = pred['name']
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        # Save
                        out_name = f"{sample_token}_{cam_name}.jpg"
                        cv2.imwrite(os.path.join(VIS_OUT_DIR, out_name), img)

        except Exception as e:
            print(f"Error processing frame {frame.get('frame_id', '???')}: {e}")
            import traceback
            traceback.print_exc()
            pass

    # Print Table
    print(f"{'Class':<25} | {'Total TP':<10} | {'Total DT':<10} | {'Total GT':<10} | {'Recall':<10} | {'Avg IoU (TPs)':<15}")
    print("-" * 95)
    
    for cls in valid_pred_classes:
        tp = metrics[cls]['TP']
        dt = metrics[cls]['DT']
        gt = metrics[cls]['GT']
        recall = tp / gt if gt > 0 else 0
        avg_iou = metrics[cls]['IoU_Sum'] / tp if tp > 0 else 0
        
        print(f"{cls:<25} | {tp:<10} | {dt:<10} | {gt:<10} | {recall:<10.4f} | {avg_iou:<15.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Eval 2D detection on NuScenes')
    parser.add_argument('--pkl', type=str, required=True, help='Path to prediction pkl file')
    parser.add_argument('--nusc_root', type=str, default='/home/ys/MS3D/data/nuscenes/v1.0-mini', help='NuScenes root directory')
    parser.add_argument('--version', type=str, default='v1.0-mini', help='NuScenes version')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)
    eval_2d(args.pkl, nusc)
