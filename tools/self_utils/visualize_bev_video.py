import pickle
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append('/MS3D')
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import build_dataloader
from pcdet.utils import box_fusion_utils
from pcdet.utils import compatibility_utils as compat
from visual_utils import common_vis
import argparse
import matplotlib
matplotlib.use('TkAgg')  # 確保使用 TkAgg 後端
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_boxes(ax, boxes_lidar, color=[0,0,1], scores=None, label=None, cur_id=0, limit_range=None,
               source_id=None, source_labels=None, alpha=1.0, linestyle='solid', linewidth=1.0, fontsize=12,
               show_score=True):
    if limit_range is not None:
        centroids = boxes_lidar[:, :3]
        mask = common_vis.mask_points_by_range(centroids, limit_range)
        boxes_lidar = boxes_lidar[mask]
        if source_labels is not None:
            source_labels = source_labels[mask]
        if source_id is not None:
            source_id = source_id[mask]
        if scores is not None:
            scores = scores[mask]

    box_pts = common_vis.boxes_to_corners_3d(boxes_lidar)
    box_pts_bev = box_pts[:, :5, :2]
    cmap = np.array(plt.get_cmap('tab20').colors)
    prev_id = -1
    for idx, box in enumerate(box_pts_bev):
        if source_id is not None:
            cur_id = source_id[idx]
            color = cmap[cur_id % len(cmap)]
            label = None
            if source_labels is not None:
                label = source_labels[idx]
        
        if cur_id != prev_id:
            ax.plot(box[:, 0], box[:, 1], color=color, label=label, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            prev_id = cur_id
        else:
            ax.plot(box[:, 0], box[:, 1], color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        if (scores is not None) and show_score:
            ax.text(box[0, 0], box[0, 1], f'{scores[idx]:0.4f}', c=color, fontsize=fontsize)

def get_frame_id_from_dets(frame_id, detection_sets):
    for i, dset in enumerate(detection_sets):
        if dset['frame_id'] == frame_id:
            return i
    return None

def main():
    parser = argparse.ArgumentParser(description='生成 BEV 視覺化影片')
    parser.add_argument('--cfg_file', type=str, default='/MS3D/tools/cfgs/dataset_configs/waymo_dataset_da.yaml',
                        help='目標數據集的配置文件')
    parser.add_argument('--dets_txt', type=str, default=None, required=False,
                        help='檢測器 pkl 路徑的 txt 文件')
    parser.add_argument('--det_pkl', type=str, default=None, required=False,
                        help='來自 test.py 的 result.pkl 文件，假設框在地面坐標系')
    parser.add_argument('--det_pkl2', type=str, default=None, required=False,
                        help='用於比較的另一個 result.pkl 文件')
    parser.add_argument('--ps_pkl', type=str, required=False,
                        help='MS3D 生成的 ps_dict_*, ps_label_e*.pkl 文件')
    parser.add_argument('--ps_pkl2', type=str, required=False,
                        help='用於比較的另一個 ps_dict 文件')
    parser.add_argument('--tracks_pkl', type=str, required=False,
                        help='追蹤數據，鍵為 ID 的字典')
    parser.add_argument('--tracks_pkl2', type=str, required=False,
                        help='用於比較的另一個追蹤數據')
    parser.add_argument('--idx', type=int, default=0,
                        help='起始幀索引')
    parser.add_argument('--sweeps', type=int, default=None,
                        help='累積點雲數量')
    parser.add_argument('--conf_th', type=float, default=0.0,
                        help='檢測置信度閾值')
    parser.add_argument('--split', type=str, default='train',
                        help='指定訓練或測試分割')
    parser.add_argument('--show_trk_score', action='store_true', default=False,
                        help='顯示追蹤分數')
    parser.add_argument('--hide_score', action='store_true', default=False,
                        help='隱藏分數標籤')
    parser.add_argument('--custom_train_split', action='store_true', default=False,
                        help='使用自定義訓練分割')
    parser.add_argument('--above_pos_th', action='store_true', default=False,
                        help='僅顯示正向置信度的框')
    parser.add_argument('--color_height', action='store_true', default=False,
                        help='按高度著色點雲')
    parser.add_argument('--frame2box_key', type=str, required=False, default=None,
                        help='選項：frameid_to_box, frameid_to_rollingkde, frameid_to_propboxes')
    parser.add_argument('--output_video', type=str, default='output.mp4',
                        help='輸出影片文件路徑')
    parser.add_argument('--codec', type=str, default='libx264',
                        help='影片編碼格式（例如 libx264, mpeg4）')
    args = parser.parse_args()

    # 載入數據集
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.DATA_SPLIT.test = args.split
    if cfg.get('SAMPLED_INTERVAL', False):
        cfg.SAMPLED_INTERVAL.test = 1
    if args.custom_train_split:
        cfg.USE_CUSTOM_TRAIN_SCENES = True

    if args.sweeps is not None:
        cfg.MAX_SWEEPS = args.sweeps

    logger.info("正在載入數據集...")
    target_set, _, _ = build_dataloader(
        dataset_cfg=cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=1, logger=logger, training=False, dist=False, workers=1
    )
    idx_to_frameid = {v: k for k, v in target_set.frameid_to_idx.items()}
    logger.info(f"數據集載入完成，共 {len(idx_to_frameid)} 幀")

    # 載入數據
    start_idx = args.idx
    get_frame_id_from_ps = False
    start_frame_id = None
    ps_dict = None
    ps_dict2 = None
    tracks = None
    tracks2 = None
    detection_sets = None
    det2 = None

    if args.ps_pkl is not None:
        with open(args.ps_pkl, 'rb') as f:
            ps_dict = pickle.load(f)
        ps_frame_ids = list(ps_dict.keys())
        start_frame_id = ps_frame_ids[start_idx]
        logger.info(f"載入 ps_pkl，共 {len(ps_frame_ids)} 幀")
    if args.ps_pkl2 is not None:
        with open(args.ps_pkl2, 'rb') as f:
            ps_dict2 = pickle.load(f)
        logger.info("載入 ps_pkl2")
    if args.tracks_pkl is not None:
        with open(args.tracks_pkl, 'rb') as f:
            tracks = pickle.load(f)
        logger.info("載入 tracks_pkl")
    if args.tracks_pkl2 is not None:
        with open(args.tracks_pkl2, 'rb') as f:
            tracks2 = pickle.load(f)
        logger.info("載入 tracks_pkl2")
    if args.dets_txt is not None:
        det_annos = box_fusion_utils.load_src_paths_txt(args.dets_txt)
        detection_sets = box_fusion_utils.get_detection_sets(det_annos, score_th=0.1)
        if args.ps_pkl is None:
            start_frame_id = detection_sets[start_idx]['frame_id']
        else:
            get_frame_id_from_ps = True
        logger.info(f"載入 dets_txt，共 {len(detection_sets)} 幀")
    if args.det_pkl is not None:
        with open(args.det_pkl, 'rb') as f:
            detection_sets = pickle.load(f)
        if args.ps_pkl is None:
            start_frame_id = detection_sets[start_idx]['frame_id']
        else:
            get_frame_id_from_ps = True
        logger.info(f"載入 det_pkl，共 {len(detection_sets)} 幀")
    if args.det_pkl2 is not None:
        with open(args.det_pkl2, 'rb') as f:
            det2 = pickle.load(f)
        logger.info("載入 det_pkl2")

    if start_frame_id is None:
        start_frame_id = idx_to_frameid[0]
        logger.info(f"使用預設起始幀 ID: {start_frame_id}")

    # 初始化圖表
    fig = plt.figure(figsize=(10, 10))  # 減小尺寸以提高性能
    ax = plt.subplot(111)
    fig.subplots_adjust(right=0.7)
    pcr = 150
    limit_range = [-pcr, -pcr, -4.0, pcr, pcr, 2.0]

    # 動畫更新函數
    def update(frame_idx):
        ax.clear()
        if args.ps_pkl is not None:
            frame_id = ps_frame_ids[frame_idx % len(ps_frame_ids)]
        elif detection_sets is not None:
            frame_id = detection_sets[frame_idx % len(detection_sets)]['frame_id']
        else:
            frame_id = idx_to_frameid[frame_idx % len(idx_to_frameid.keys())]

        if frame_id not in target_set.frameid_to_idx:
            logger.warning(f"幀 ID {frame_id} 在數據集中不存在，跳過")
            return

        pts = target_set[target_set.frameid_to_idx[frame_id]]['points']
        mask = common_vis.mask_points_by_range(pts, limit_range)
        pts = pts[mask]

        if args.color_height:
            cmap = plt.get_cmap('gray')
            normed_z = (pts[:, 2] - min(pts[:, 2])) / (max(pts[:, 2]) - min(pts[:, 2]))
            mapped_z = 1 / (1 + np.exp(-5 * (normed_z - 0.5)))
            mapped_z = np.clip(mapped_z + 0.35, 0, 1)
            z_colors = np.array([cmap(1 - n_z) for n_z in mapped_z])
            ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c=z_colors, marker='o')
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c='black', marker='o')

        # 繪製真值框
        class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Vehicle', 'car', 'truck', 'bus'])
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0, 0, 1],
                   limit_range=limit_range, label='真值_車輛', linewidth=2,
                   scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]),
                   show_score=False if args.hide_score else True)

        class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Pedestrian', 'pedestrian'])
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0.5, 0, 0.5],
                   limit_range=limit_range, label='真值_行人', linewidth=2,
                   scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]),
                   show_score=False if args.hide_score else True)

        class_mask = np.isin(compat.get_gt_names(target_set, frame_id), ['Cyclist', 'bicycle', 'motorcycle'])
        plot_boxes(ax, compat.get_gt_boxes(target_set, frame_id)[class_mask], color=[0, 0.5, 0.5],
                   limit_range=limit_range, label='真值_騎行者',
                   scores=np.ones(compat.get_gt_boxes(target_set, frame_id)[class_mask].shape[0]),
                   show_score=False if args.hide_score else True)

        # 繪製檢測框
        if detection_sets is not None:
            det_frame_idx = get_frame_id_from_dets(frame_id, detection_sets) if get_frame_id_from_ps else frame_idx
            if det_frame_idx is not None:
                conf_mask = detection_sets[det_frame_idx]['score'] >= args.conf_th
                plot_boxes(ax, detection_sets[det_frame_idx]['boxes_lidar'][conf_mask],
                           scores=detection_sets[det_frame_idx]['score'][conf_mask],
                           source_id=detection_sets[det_frame_idx]['source_id'][conf_mask] if 'source_id' in detection_sets[det_frame_idx].keys() else None,
                           source_labels=detection_sets[det_frame_idx]['source'][conf_mask] if 'source' in detection_sets[det_frame_idx].keys() else None,
                           color=[0, 0.8, 0] if 'source_id' not in detection_sets[det_frame_idx].keys() else [0, 0, 1],
                           limit_range=limit_range, alpha=0.5 if 'source_id' in detection_sets[det_frame_idx].keys() else 1.0,
                           label='檢測框' if 'source_id' not in detection_sets[det_frame_idx].keys() else None,
                           show_score=False if args.hide_score else True)

                if args.det_pkl2 is not None:
                    conf_mask = det2[det_frame_idx]['score'] >= args.conf_th
                    plot_boxes(ax, det2[det_frame_idx]['boxes_lidar'][conf_mask],
                               scores=det2[det_frame_idx]['score'][conf_mask],
                               label='檢測框 2', color=[0.6, 0.4, 0],
                               limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)

        # 繪製偽標籤
        if args.ps_pkl is not None:
            combined_mask = ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][:, 8] >= args.conf_th
            if args.above_pos_th:
                above_pos_mask = ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][:, 7] > 0
                combined_mask = np.logical_and(combined_mask, above_pos_mask)
            plot_boxes(ax, ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask],
                       scores=ps_dict[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask][:, 8],
                       label='偽標籤', color=[0, 0.8, 0], fontsize=14, linewidth=1.5,
                       limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)

        if args.ps_pkl2 is not None:
            combined_mask = ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][:, 8] >= args.conf_th
            if args.above_pos_th:
                above_pos_mask = ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][:, 7] > 0
                combined_mask = np.logical_and(combined_mask, above_pos_mask)
            plot_boxes(ax, ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask],
                       scores=ps_dict2[ps_frame_ids[frame_idx]]['gt_boxes'][combined_mask][:, 8],
                       label='偽標籤 2', color=[1, 0, 0],
                       limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)

        # 繪製追蹤框
        if args.tracks_pkl is not None:
            from pcdet.utils.transform_utils import world_to_ego
            from pcdet.utils.tracker_utils import get_frame_track_boxes
            track_boxes = get_frame_track_boxes(tracks, frame_id, frame2box_key=args.frame2box_key, nhistory=0)
            pose = compat.get_pose(target_set, frame_id)
            _, track_boxes_ego = world_to_ego(pose, boxes=track_boxes)
            if track_boxes_ego.shape[0] != 0:
                score_idx = 7 if args.show_trk_score else 8
                plot_boxes(ax, track_boxes_ego[:, :7],
                           scores=track_boxes_ego[:, score_idx],
                           label='追蹤框', color=[1, 0, 0], linestyle='dotted',
                           limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)

        if args.tracks_pkl2 is not None:
            track_boxes2 = get_frame_track_boxes(tracks2, frame_id, nhistory=0)
            pose = compat.get_pose(target_set, frame_id)
            _, track_boxes_ego2 = world_to_ego(pose, boxes=track_boxes2)
            if track_boxes_ego2.shape[0] != 0:
                score_idx = 7 if args.show_trk_score else 8
                plot_boxes(ax, track_boxes_ego2[:, :7],
                           scores=track_boxes_ego2[:, score_idx],
                           label='追蹤框 2', color=[1, 0.7, 0], linestyle='dotted',
                           limit_range=limit_range, alpha=1, show_score=False if args.hide_score else True)

        # 設置標題
        if 'scene_name' in target_set.infos[target_set.frameid_to_idx[frame_id]]:
            scene_name = target_set.infos[target_set.frameid_to_idx[frame_id]]['scene_name']
            ax.set_title(f'幀 #{frame_idx}, 場景: {scene_name}, FID: {frame_id}')
        else:
            ax.set_title(f'幀 #{frame_idx}, FID: {frame_id}')

        ax.set_xlim(0, 150)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0, 0.07, 1))
        logger.info(f"渲染幀 {frame_idx}")

    # 確定總幀數
    if args.ps_pkl is not None:
        total_frames = len(ps_frame_ids)
    elif detection_sets is not None:
        total_frames = len(detection_sets)
    else:
        total_frames = len(idx_to_frameid.keys())
    logger.info(f"總共需渲染 {total_frames} 幀")

    # 創建動畫
    try:
        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=200, blit=False)
        writer = animation.FFMpegWriter(fps=5, bitrate=5000, codec=args.codec, extra_args=['-loglevel', 'verbose'])
        ani.save(args.output_video, writer=writer)
        logger.info(f"影片已保存為 {args.output_video}")
    except Exception as e:
        logger.error(f"保存影片失敗: {str(e)}")
        raise

    # 關閉圖表以釋放記憶體
    plt.close(fig)

if __name__ == '__main__':
    main()