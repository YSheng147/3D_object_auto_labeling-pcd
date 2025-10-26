#!/bin/bash

# We set sweeps based on the assumption that your lidar data is at 10Hz
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/nusc_pt_dets_uda_pv_a_10xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/nusc_pt_dets_uda_pv_c_10xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/nusc_pt_dets_uda_vx_a_10xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/nusc_pt_dets_uda_vx_c_10xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/waymo_pt_dets_uda_pv_a_4xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/waymo_pt_dets_uda_pv_c_4xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/waymo_pt_dets_uda_vx_a_4xyzt.sh
bash ~/MS3D/tools/cfgs/target_custom/label_generation/round1/scripts/pretrained/waymo_pt_dets_uda_vx_c_4xyzt.sh