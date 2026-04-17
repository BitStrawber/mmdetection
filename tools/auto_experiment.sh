#!/bin/bash
# =============================================================================
# auto_experiment.sh - 自动化UWNR转换 + 实验A/B全流程
#
# 重要：使用前请修改下方的"配置区域"路径
#
# 
# 功能：
#   1. 监控UWNR转换进度（使用GPU 6,7）
#   2. UWNR完成后自动运行实验A（ImageNet+RUOD）
#   3. 实验A完成后自动运行实验B-1（COCO-UWNR预训练）
#   4. 实验B-1完成后自动提取backbone权重
#   5. 实验B-3（COCO-UWNR+RUOD微调）
#   6. 所有实验完成后的结果整理
#
# 用法：
#   chmod +x auto_experiment.sh
#   nohup bash auto_experiment.sh > auto_experiment.log 2>&1 &
#
# 前置条件（请确保以下路径正确）：
#   - UWNR环境已配置
#   - mmdetection环境已配置
#   - COCO数据集已准备
# =============================================================================

# ======================== 配置区域 ========================
# 项目路径
UWNR_DIR="/home/fcp/xcx/UWNR"
MMDET_DIR="/home/fcp/xcx/mmdetection"
WORK_DIR="/home/fcp/xcx/experiment_workspace"

# 数据路径
COCO_ANN="/media/HDD0/XCX/COCO/annotations/instances_train2017.json"
COCO_IMG="/media/HDD0/XCX/COCO/train2017"
COCO_UWNR_DIR="/media/HDD0/XCX/COCO/coco_uwnr"
COCO_CONVERT_DIR="/media/HDD0/XCX/COCO/coco_convert_uwnr"
UWNR_MODEL="/home/fcp/xcx/UWNR/UWNR.pk"

# 环境名称
UWNR_ENV="uwnr"
MMDET_ENV="mmdet"

# GPU配置
GPU_UWNR="6"           # UWNR转换使用单卡（省显存）
GPU_TRAIN="6,7"        # 训练使用双卡

# 日志文件
LOG_FILE="${WORK_DIR}/auto_experiment.log"
PID_FILE="${WORK_DIR}/auto_experiment.pid"
STATUS_FILE="${WORK_DIR}/experiment_status.json"

# 创建workspace目录
mkdir -p "${WORK_DIR}"

# ======================== Conda初始化 ========================
# 自动检测conda路径并初始化
if [ -z "$CONDA_EXE" ]; then
    # 尝试常见conda安装位置
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/media/SSD1/conda_envs/etc/profile.d/conda.sh" ]; then
        source "/media/SSD1/conda_envs/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    else
        log "错误: 无法找到conda安装，请手动修改脚本添加conda.sh路径"
        exit 1
    fi
fi

# 激活环境的辅助函数
activate_env() {
    local env_name=$1
    log "激活conda环境: ${env_name}"
    conda activate "${env_name}" || {
        log "错误: 无法激活环境 ${env_name}"
        exit 1
    }
}
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/results"

# ======================== 工具函数 ========================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

save_status() {
    echo "$1" > "${STATUS_FILE}"
}

get_status() {
    if [ -f "${STATUS_FILE}" ]; then
        cat "${STATUS_FILE}"
    else
        echo "INIT"
    fi
}

# 检查UWNR转换是否完成
check_uwnr_complete() {
    local total_images=50000
    local output_dir="${COCO_CONVERT_DIR}/images"
    
    if [ ! -d "$output_dir" ]; then
        return 1
    fi
    
    local current_count=$(ls -1 "$output_dir" 2>/dev/null | wc -l)
    log "UWNR进度: ${current_count}/${total_images}"
    
    if [ "$current_count" -ge "$total_images" ]; then
        return 0
    else
        return 1
    fi
}

# 检查进程是否在运行
is_process_running() {
    pgrep -f "$1" > /dev/null 2>&1
}

# 等待GPU空闲（显存<500MB）
wait_gpu_free() {
    local gpu_id=$1
    local threshold=500
    
    while true; do
        local mem_used=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
        if [ "$mem_used" -lt "$threshold" ]; then
            log "GPU ${gpu_id} 已空闲 (${mem_used}MB)"
            return 0
        fi
        log "GPU ${gpu_id} 占用 ${mem_used}MB，等待中..."
        sleep 60
    done
}

# ======================== 阶段1: 监控UWNR转换 ========================
wait_uwnr_complete() {
    log "========== 阶段1: 监控UWNR数据转换 =========="
    
    # 检查是否已完成
    if check_uwnr_complete; then
        log "UWNR转换已完成"
        save_status "UWNRDONE"
        return 0
    fi
    
    log "请手动启动UWNR转换（使用GPU 6,7），脚本将自动检测完成状态"
    log ""
    log "手动启动命令示例（在UWNR目录下，激活uwnr环境后执行）："
    log "  # 进程1: GPU 6, 前25000张"
    log "  CUDA_VISIBLE_DEVICES=6 python convert_coco_uwnr.py \\"
    log "      --ann ${COCO_UWNR_DIR}/annotations/instances_train50000.json \\"
    log "      --img-dir ${COCO_UWNR_DIR}/images \\"
    log "      --output-dir ${COCO_CONVERT_DIR} \\"
    log "      --uwnr-dir ${UWNR_DIR} \\"
    log "      --uwnr-model ${UWNR_MODEL} \\"
    log "      --gpu 6 --start 0 --end 25000 &"
    log ""
    log "  # 进程2: GPU 7, 后25000张"
    log "  CUDA_VISIBLE_DEVICES=7 python convert_coco_uwnr.py \\"
    log "      --ann ${COCO_UWNR_DIR}/annotations/instances_train50000.json \\"
    log "      --img-dir ${COCO_UWNR_DIR}/images \\"
    log "      --output-dir ${COCO_CONVERT_DIR} \\"
    log "      --uwnr-dir ${UWNR_DIR} \\"
    log "      --uwnr-model ${UWNR_MODEL} \\"
    log "      --gpu 7 --start 25000 --end 50000 &"
    log ""
    
    # 持续监控进度
    log "等待UWNR转换完成..."
    local check_count=0
    while true; do
        sleep 60  # 每分钟检查一次
        check_count=$((check_count + 1))
        
        local current_count=$(ls -1 "${COCO_CONVERT_DIR}/images" 2>/dev/null | wc -l)
        
        # 每10分钟显示一次详细进度
        if [ $((check_count % 10)) -eq 0 ]; then
            log "UWNR进度: ${current_count}/50000 (已等待 $((check_count/60)) 小时)"
        fi
        
        if check_uwnr_complete; then
            log "UWNR转换完成！共转换 50000 张图片"
            save_status "UWNRDONE"
            return 0
        fi
    done
}

# ======================== 阶段2: 实验A ========================
run_experiment_a() {
    log "========== 阶段2: 实验A (ImageNet+RUOD) =========="
    
    cd "${MMDET_DIR}" || exit 1
    activate_env "${MMDET_ENV}"
    
    # 等待GPU空闲
    log "等待GPU ${GPU_TRAIN} 空闲..."
    for gpu in $(echo $GPU_TRAIN | tr ',' ' '); do
        wait_gpu_free $gpu
    done
    
    log "开始实验A训练..."
    CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" \
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=29500 \
        tools/train.py \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py \
        2>&1 | tee "${WORK_DIR}/logs/04_expA_train.log"
    
    local exit_code=$?
    conda deactivate || true
    
    if [ $exit_code -eq 0 ]; then
        log "实验A训练完成"
        
        # 复制权重到结果目录
        local expA_weight="${MMDET_DIR}/work_dirs/cascade-rcnn_r50_fpn_2x_ruod/latest.pth"
        if [ -f "$expA_weight" ]; then
            cp "$expA_weight" "${WORK_DIR}/results/expA_final.pth"
            log "实验A权重已保存"
        fi
        
        # 运行测试
        log "开始实验A测试..."
        activate_env "${MMDET_ENV}"
        CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" \
        python tools/test.py \
            configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod.py \
            "${expA_weight}" \
            --eval bbox \
            --cfg-options model.init_cfg=None \
            2>&1 | tee "${WORK_DIR}/logs/05_expA_test.log"
        conda deactivate || true
        
        save_status "EXPADONE"
        return 0
    else
        log "实验A训练失败"
        return 1
    fi
}

# ======================== 阶段3: 实验B-1 ========================
run_experiment_b1() {
    log "========== 阶段3: 实验B-1 (COCO-UWNR预训练) =========="
    
    cd "${MMDET_DIR}" || exit 1
    activate_env "${MMDET_ENV}"
    
    # 等待GPU空闲
    for gpu in $(echo $GPU_TRAIN | tr ',' ' '); do
        wait_gpu_free $gpu
    done
    
    log "开始实验B-1训练..."
    CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" \
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=29501 \
        tools/train.py \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_coco_uwnr.py \
        2>&1 | tee "${WORK_DIR}/logs/06_expB1_train.log"
    
    local exit_code=$?
    conda deactivate || true
    
    if [ $exit_code -eq 0 ]; then
        log "实验B-1训练完成"
        
        # 复制权重
        local expB1_weight="${MMDET_DIR}/work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/latest.pth"
        if [ -f "$expB1_weight" ]; then
            cp "$expB1_weight" "${WORK_DIR}/results/expB1_final.pth"
            log "实验B-1权重已保存"
        fi
        
        save_status "EXPB1DONE"
        return 0
    else
        log "实验B-1训练失败"
        return 1
    fi
}

# ======================== 阶段4: 提取Backbone ========================
extract_backbone() {
    log "========== 阶段4: 提取Backbone权重 =========="
    
    cd "${MMDET_DIR}" || exit 1
    activate_env "${MMDET_ENV}"
    
    local input_weight="${MMDET_DIR}/work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr/latest.pth"
    local output_dir="${MMDET_DIR}/work_dirs/cascade-rcnn_r50_fpn_2x_coco_uwnr"
    
    if [ ! -f "$input_weight" ]; then
        log "错误: 实验B-1权重不存在"
        return 1
    fi
    
    log "提取backbone权重..."
    python tools/extract_backbone.py \
        --checkpoint "$input_weight" \
        --output "${output_dir}/backbone_only.pth" \
        2>&1 | tee "${WORK_DIR}/logs/07_extract_backbone.log"
    
    if [ -f "${output_dir}/backbone_only.pth" ]; then
        cp "${output_dir}/backbone_only.pth" "${WORK_DIR}/results/backbone_only.pth"
        log "Backbone权重已保存"
        save_status "BACKBONEEXTRACTED"
        return 0
    else
        log "Backbone提取失败"
        return 1
    fi
}

# ======================== 阶段5: 实验B-3 ========================
run_experiment_b3() {
    log "========== 阶段5: 实验B-3 (COCO-UWNR+RUOD微调) =========="
    
    cd "${MMDET_DIR}" || exit 1
    activate_env "${MMDET_ENV}"
    
    # 等待GPU空闲
    for gpu in $(echo $GPU_TRAIN | tr ',' ' '); do
        wait_gpu_free $gpu
    done
    
    log "开始实验B-3训练..."
    CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" \
    python -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=29502 \
        tools/train.py \
        configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain.py \
        2>&1 | tee "${WORK_DIR}/logs/08_expB3_train.log"
    
    local exit_code=$?
    conda deactivate || true
    
    if [ $exit_code -eq 0 ]; then
        log "实验B-3训练完成"
        
        # 复制权重
        local expB3_weight="${MMDET_DIR}/work_dirs/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain/latest.pth"
        if [ -f "$expB3_weight" ]; then
            cp "$expB3_weight" "${WORK_DIR}/results/expB3_final.pth"
            log "实验B-3权重已保存"
        fi
        
        # 运行测试
        log "开始实验B-3测试..."
        activate_env "${MMDET_ENV}"
        CUDA_VISIBLE_DEVICES="${GPU_TRAIN}" \
        python tools/test.py \
            configs/cascade_rcnn/cascade-rcnn_r50_fpn_2x_ruod_uwnr_pretrain.py \
            "${expB3_weight}" \
            --eval bbox \
            --cfg-options model.init_cfg=None \
            2>&1 | tee "${WORK_DIR}/logs/09_expB3_test.log"
        conda deactivate || true
        
        save_status "EXPB3DONE"
        return 0
    else
        log "实验B-3训练失败"
        return 1
    fi
}

# ======================== 阶段6: 结果整理 ========================
summarize_results() {
    log "========== 阶段6: 结果整理 =========="
    
    local summary_file="${WORK_DIR}/results/experiment_summary.txt"
    
    cat > "$summary_file" << 'EOF'
实验完成总结
==============
实验日期: $(date)

实验配置:
- GPU: 6,7
- Batch Size: 6/GPU (总BS=12 for 2 GPUs)
- Learning Rate: 0.015
- Epoch: 24
- MultiStep: [16, 22]

权重文件:
- 实验A (ImageNet+RUOD): expA_final.pth
- 实验B-1 (COCO-UWNR): expB1_final.pth
- 提取的Backbone: backbone_only.pth
- 实验B-3 (COCO-UWNR+RUOD): expB3_final.pth

日志文件位置: ${WORK_DIR}/logs/
EOF

    log "结果汇总已保存到: $summary_file"
    
    # 生成权重文件列表
    ls -lh "${WORK_DIR}/results/" > "${WORK_DIR}/results/weight_files.txt"
    
    save_status "ALLDONE"
    log "========== 所有实验完成！=========="
}

# ======================== 主程序 ========================
main() {
    log "========================================"
    log "自动化实验脚本启动"
    log "工作目录: ${WORK_DIR}"
    log "GPU配置: Train=${GPU_TRAIN}, UWNR=${GPU_UWNR}"
    log "========================================"
    
    # 保存PID
    echo $$ > "${PID_FILE}"
    
    # 获取当前状态
    local status=$(get_status)
    log "当前状态: ${status}"
    
    # 状态机
    case "$status" in
        "INIT"|"")
            wait_uwnr_complete && run_experiment_a && run_experiment_b1 && extract_backbone && run_experiment_b3 && summarize_results
            ;;
        "UWNRDONE")
            run_experiment_a && run_experiment_b1 && extract_backbone && run_experiment_b3 && summarize_results
            ;;
        "EXPADONE")
            run_experiment_b1 && extract_backbone && run_experiment_b3 && summarize_results
            ;;
        "EXPB1DONE")
            extract_backbone && run_experiment_b3 && summarize_results
            ;;
        "BACKBONEEXTRACTED")
            run_experiment_b3 && summarize_results
            ;;
        "EXPB3DONE")
            summarize_results
            ;;
        *)
            log "未知状态: ${status}，从头开始"
            wait_uwnr_complete && run_experiment_a && run_experiment_b1 && extract_backbone && run_experiment_b3 && summarize_results
            ;;
    esac
    
    # 清理
    rm -f "${PID_FILE}"
}

# 捕获信号
trap 'log "脚本被中断"; rm -f "${PID_FILE}"; exit 1' INT TERM

# 运行
main "$@"
