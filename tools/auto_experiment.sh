#!/bin/bash
# =============================================================================
# auto_experiment.sh - 自动化实验全流程
#
# 前置: UWNR转换已手动启动，脚本监控完成后自动运行后续实验
#
# 用法:
#   nohup bash tools/auto_experiment.sh > auto_experiment.log 2>&1 &
#   tail -f auto_experiment.log
# =============================================================================

# ======================== 配置区域 ========================
WORK_DIR="/home/fcp/xcx/experiment_workspace"
COCO_CONVERT_DIR="/media/HDD0/XCX/COCO/coco_convert_uwnr"
MMDET_DIR="/home/fcp/xcx/mmdetection"
RUOD_ANN="/media/HDD0/XCX/RUOD/RUOD_ANN"
NUM_GPUS=2

LOG_DIR="${WORK_DIR}/logs"
RESULT_DIR="${WORK_DIR}/results"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${WORK_DIR}/auto_experiment.log"
}

# ======================== 阶段1: 监控UWNR转换 ========================
wait_uwnr_complete() {
    log "========== 阶段1: 监控UWNR数据转换 =========="
    local img_dir="${COCO_CONVERT_DIR}/images"
    local total=50000

    if [ ! -d "$img_dir" ]; then
        log "输出目录不存在: ${img_dir}"
        log "请先手动启动UWNR转换（2个终端）:"
        log "  cd /home/fcp/xcx/UWNR && conda activate uwnr"
        log "  CUDA_VISIBLE_DEVICES=6 python convert_coco_uwnr.py --ann ... --uw-img-dir /media/HDD0/XCX/RUOD/RUOD_pic/train --gpu 6 --start 0 --end 25000"
        log "  CUDA_VISIBLE_DEVICES=7 python convert_coco_uwnr.py --ann ... --uw-img-dir /media/HDD0/XCX/RUOD/RUOD_pic/train --gpu 7 --start 25000 --end 50000"
    fi

    while true; do
        local count=$(ls -1 "$img_dir" 2>/dev/null | wc -l)
        if [ "$count" -ge "$total" ]; then
            log "UWNR转换完成! 共 ${count} 张"
            return 0
        fi
        log "UWNR进度: ${count}/${total}"
        sleep 60
    done
}

# ======================== 阶段2: 实验A (ImageNet+RUOD) ========================
run_experiment_a() {
    log "========== 阶段2: 实验A (ImageNet+RUOD baseline) =========="
    cd "${MMDET_DIR}" || exit 1

    bash tools/dist_train.sh \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py \
        ${NUM_GPUS} \
        --work-dir work_dirs/cascade-rcnn_r50_fpn_2x_ruod \
        2>&1 | tee "${LOG_DIR}/02_expA_train.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "实验A训练失败"; return 1
    fi
    log "实验A训练完成"

    cp work_dirs/cascade-rcnn_r50_fpn_2x_ruod/best_coco_bbox_mAP*.pth \
       "${RESULT_DIR}/expA_best.pth" 2>/dev/null || true

    log "实验A测试..."
    bash tools/dist_test.sh \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py \
        work_dirs/cascade-rcnn_r50_fpn_2x_ruod/best_coco_bbox_mAP*.pth \
        ${NUM_GPUS} \
        --eval bbox --cfg-options model.init_cfg=None \
        2>&1 | tee "${LOG_DIR}/03_expA_test.log"
    log "实验A完成"
}

# ======================== 阶段3: 实验B-1 (COCO-UWNR预训练) ========================
run_experiment_b1() {
    log "========== 阶段3: 实验B-1 (COCO-UWNR预训练) =========="
    cd "${MMDET_DIR}" || exit 1

    bash tools/dist_train.sh \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_coco_uwnr.py \
        ${NUM_GPUS} \
        --work-dir work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr \
        2>&1 | tee "${LOG_DIR}/04_expB1_train.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "实验B-1训练失败"; return 1
    fi
    log "实验B-1训练完成"

    cp work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/best_coco_bbox_mAP*.pth \
       "${RESULT_DIR}/expB1_best.pth" 2>/dev/null || true
}

# ======================== 阶段4: 提取Backbone ========================
extract_backbone_weights() {
    log "========== 阶段4: 提取Backbone权重 =========="
    cd "${MMDET_DIR}" || exit 1

    local ckpt=$(ls -t work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/best_coco_bbox_mAP*.pth 2>/dev/null | head -1)
    if [ -z "$ckpt" ]; then
        ckpt="work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/latest.pth"
    fi

    python tools/extract_backbone.py \
        --checkpoint "$ckpt" \
        --output work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/backbone_only.pth \
        2>&1 | tee "${LOG_DIR}/05_extract_backbone.log"

    if [ ! -f "work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/backbone_only.pth" ]; then
        log "Backbone提取失败"; return 1
    fi
    cp work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/backbone_only.pth "${RESULT_DIR}/backbone_only.pth"
    log "Backbone权重已提取"
}

# ======================== 阶段5: 实验B-3 (COCO-UWNR+RUOD微调) ========================
run_experiment_b3() {
    log "========== 阶段5: 实验B-3 (COCO-UWNR+RUOD微调) =========="
    cd "${MMDET_DIR}" || exit 1

    bash tools/dist_train.sh \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain.py \
        ${NUM_GPUS} \
        --work-dir work_dirs/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain \
        2>&1 | tee "${LOG_DIR}/06_expB3_train.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "实验B-3训练失败"; return 1
    fi
    log "实验B-3训练完成"

    cp work_dirs/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain/best_coco_bbox_mAP*.pth \
       "${RESULT_DIR}/expB3_best.pth" 2>/dev/null || true

    log "实验B-3测试..."
    bash tools/dist_test.sh \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain.py \
        work_dirs/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain/best_coco_bbox_mAP*.pth \
        ${NUM_GPUS} \
        --eval bbox --cfg-options model.init_cfg=None \
        2>&1 | tee "${LOG_DIR}/07_expB3_test.log"
    log "实验B-3完成"
}

# ======================== 阶段6: 结果整理 ========================
summarize() {
    log "========== 阶段6: 结果整理 =========="
    {
        echo "=============================="
        echo "实验完成总结 - $(date)"
        echo "=============================="
        echo ""
        echo "权重文件:"
        ls -lh "${RESULT_DIR}/"
        echo ""
        echo "实验A mAP:"
        grep -oP 'bbox_mAP: \K[0-9.]+' "${LOG_DIR}/03_expA_test.log" | tail -1
        echo "实验B-3 mAP:"
        grep -oP 'bbox_mAP: \K[0-9.]+' "${LOG_DIR}/07_expB3_test.log" | tail -1
    } | tee "${RESULT_DIR}/summary.txt"
    log "========== 全部完成 =========="
}

# ======================== 主程序 ========================
log "========================================"
log "自动化实验脚本启动"
log "工作目录: ${WORK_DIR}"
log "========================================"

wait_uwnr_complete && \
run_experiment_a && \
run_experiment_b1 && \
extract_backbone_weights && \
run_experiment_b3 && \
summarize

log "脚本结束"
