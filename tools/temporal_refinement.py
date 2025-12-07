"""
MS3D Step 3 (final)

DESCRIPTION:
    Temporally refines all tracks and detection sets using object characteristics.

EXAMPLES:
    python temporal_refinement.py --ps_cfg /MS3D/tools/cfgs/target_nuscenes/ms3d_ps_config_rnd3_2.yaml 
"""
import sys
sys.path.append('../')
from pcdet.utils import box_fusion_utils, ms3d_utils
import argparse
from pcdet.config import cfg, cfg_from_yaml_file
import yaml
from pathlib import Path
from self_utils.box_final_conv import *
import numpy as np


def pkl_iou_cal(cfg, gt_data, data_cfg_path):
    # open file
    with open(data_cfg_path, 'r') as f:
        pkl_pths = [line.split('\n')[0] for line in f.readlines()]

    for idx, pkl_pth_w in enumerate(pkl_pths):
        # open pkl
        split_pth_w = pkl_pth_w.split(',')
        pkl_pth = split_pth_w[0]
        pkl = ms3d_utils.load_pkl(pkl_pth) 

        print(f"\n{pkl_pth}")
        iou_cal(cfg, gt_data, pkl)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')                   
    parser.add_argument('--ps_cfg', type=str, help='cfg file with MS3D parameters')
    parser.add_argument('--save_veh_intermediate_tracks', action='store_true', default=False, help='Save static vehicle tracks at each stage: motion refinement, rolling_kde, and propagate boxes')
    args = parser.parse_args()

    ms3d_configs = yaml.load(open(args.ps_cfg,'r'), Loader=yaml.Loader)
    cfg_from_yaml_file(ms3d_configs["DATA_CONFIG_PATH"], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split='train')
    
    # Load pkls
    ps_pth = Path(ms3d_configs["SAVE_DIR"]) / f'initial_pseudo_labels.pkl'
    tracks_veh_all_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_veh_all.pkl'
    tracks_veh_static_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_veh_static.pkl'
    tracks_ped_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_ped.pkl'
    ps_dict = ms3d_utils.load_pkl(ps_pth)
    tracks_veh_all = ms3d_utils.load_pkl(tracks_veh_all_pth)
    tracks_veh_static = ms3d_utils.load_pkl(tracks_veh_static_pth)
    tracks_ped = ms3d_utils.load_pkl(tracks_ped_pth)
    # cyclist
    tracks_cyc_pth = Path(ms3d_configs["SAVE_DIR"]) / f'tracks_world_cyc.pkl'
    tracks_cyc = ms3d_utils.load_pkl(tracks_cyc_pth)

    # Get vehicle labels
    print('Refining vehicle labels')
    tracks_veh_all, tracks_veh_static = ms3d_utils.refine_veh_labels(dataset,list(ps_dict.keys()),
                                                                    tracks_veh_all, 
                                                                    tracks_veh_static, 
                                                                    static_trk_score_th=ms3d_configs['TRACKING']['VEH_STATIC']['RUNNING']['SCORE_TH'],
                                                                    veh_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][0],
                                                                    refine_cfg=ms3d_configs['TEMPORAL_REFINEMENT'],
                                                                    save_dir=ms3d_configs['SAVE_DIR'] if args.save_veh_intermediate_tracks else None)

    # Get pedestrian labels
    print('Refining pedestrian labels')
    tracks_ped = ms3d_utils.refine_ped_labels(tracks_ped, 
                                              ped_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][1],
                                              track_filtering_cfg=ms3d_configs['TEMPORAL_REFINEMENT']['TRACK_FILTERING'])

    # Get cyclist labels
    print('Refining cyclist labels')
    tracks_cyc = ms3d_utils.refine_ped_labels(tracks_cyc, 
                                              ped_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][2],
                                              track_filtering_cfg=ms3d_configs['TEMPORAL_REFINEMENT']['TRACK_FILTERING'])

    # Combine pseudo-labels for each class and filter with NMS
    print('Combining pseudo-labels for each class')
    # final_ps_dict = ms3d_utils.update_ps(dataset, ps_dict, tracks_veh_all, tracks_veh_static, tracks_ped, 
    #           veh_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][0], 
    #           veh_nms_th=0.05, ped_nms_th=0.5,
    #           frame2box_key_static='frameid_to_propboxes', 
    #           frame2box_key='frameid_to_box', frame_ids=list(ps_dict.keys()))
    final_ps_dict = ms3d_utils.update_ps(dataset, ps_dict, tracks_veh_all, tracks_veh_static, tracks_ped, tracks_cyc,
              veh_pos_th=ms3d_configs['PS_SCORE_TH']['POS_TH'][0], 
              veh_nms_th=0.05, ped_nms_th=0.5, cyc_nms_th=0.5,
              frame2box_key_static='frameid_to_propboxes', 
              frame2box_key='frameid_to_box', frame_ids=list(ps_dict.keys()))

    final_ps_dict = ms3d_utils.select_ps_by_th(final_ps_dict, ms3d_configs['PS_SCORE_TH']['POS_TH'])
    ms3d_utils.save_data(final_ps_dict, str(Path(ms3d_configs["SAVE_DIR"])), name="final_ps_dict.pkl")

    print('Finished generating pseudo-labels')

    final_ps_dict_conv = convert_data_format(cfg.DATASET, dataset, final_ps_dict)
    ms3d_utils.save_data(final_ps_dict_conv, str(Path(ms3d_configs["SAVE_DIR"])), name="final_ps_dict_conv.pkl")
    print('Finished converting pseudo-labels')
    
    #gt_data = get_box_name(dataset)
    #iou_cal(cfg, gt_data, final_ps_dict_conv)

    #pkl_iou_cal(cfg, gt_data, ms3d_configs['DETS_TXT'])
    