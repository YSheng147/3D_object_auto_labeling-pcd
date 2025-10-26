#!/bin/bash


CFG_FILE="../model_zoo/waymo/cfgs/waymo_uda_pv_rcnn_plusplus_resnet_anchorhead_4xyzt_allcls.yaml"
CKPT_FILE="../model_zoo/waymo/waymo_uda_pv_rcnn_plusplus_resnet_anchorhead_4xyzt_allcls.pth"

python test.py --cfg_file $CFG_FILE \
                --ckpt $CKPT_FILE \
                --eval_tag waymo4xyzt_custom1xyzt_notta \
                --target_dataset custom --sweeps 1 --batch_size 8 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file $CFG_FILE \
                --ckpt $CKPT_FILE \
                --eval_tag waymo4xyzt_custom1xyzt_rwf_rwr \
                --target_dataset custom --sweeps 1 --batch_size 8 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none 

python test.py --cfg_file $CFG_FILE \
                --ckpt $CKPT_FILE \
                --eval_tag waymo4xyzt_custom2xyzt_notta \
                --target_dataset custom --sweeps 2 --batch_size 8 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file $CFG_FILE \
                --ckpt $CKPT_FILE \
                --eval_tag waymo4xyzt_custom2xyzt_rwf_rwr \
                --target_dataset custom --sweeps 2 --batch_size 8 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                

python test.py --cfg_file $CFG_FILE \
                --ckpt $CKPT_FILE \
                --eval_tag waymo4xyzt_custom4xyzt_notta \
                --target_dataset custom --sweeps 4 --batch_size 8 --use_tta 0 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none

python test.py --cfg_file $CFG_FILE \
                --ckpt $CKPT_FILE \
                --eval_tag waymo4xyzt_custom4xyzt_rwf_rwr \
                --target_dataset custom --sweeps 4 --batch_size 8 --use_tta 3 \
                --set DATA_CONFIG_TAR.DATA_SPLIT.test train MODEL.POST_PROCESSING.EVAL_METRIC none                                                             

# ---------------------