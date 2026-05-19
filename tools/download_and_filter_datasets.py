#!/usr/bin/env python3
"""
下载并筛选4个水下数据集 (最大bbox >= 20% 图片面积)
保存到 /media/HDD1/XCX/exp_2/

数据集:
  1. UOT100    - Kaggle (单目标跟踪, 1类)
  2. UVOT400   - Google Drive (单目标跟踪, 1类)
  3. USIS16K   - Google Drive (显著性实例分割, 158类)
  4. CoralSCOP - 珊瑚分割数据集

用法:
  python tools/download_and_filter_datasets.py                     # 全部处理
  python tools/download_and_filter_datasets.py --datasets uot100   # 只处理UOT100
  python tools/download_and_filter_datasets.py --download-only     # 仅下载不筛选
"""
import os, sys, json, glob, argparse, subprocess, shutil
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = '/media/HDD1/XCX/exp_2'
THRESHOLD = 0.2


def log(msg):
    print(f"[INFO] {msg}")


def run_cmd(cmd, cwd=None, timeout=7200):
    log(f"运行: {cmd[:200]}")
    try:
        ret = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                             text=True, timeout=timeout)
        if ret.returncode != 0:
            log(f"  失败: {ret.stderr.strip()[-300:]}")
            return False
        if ret.stdout.strip():
            log(f"  {ret.stdout.strip()[-200:]}")
        return True
    except subprocess.TimeoutExpired:
        log(f"  超时")
        return False


# ====================== 筛选工具 ======================

def filter_coco(coco, threshold=THRESHOLD):
    """筛选最大bbox >= threshold * 图片面积 的图片"""
    img_map = {i['id']: i for i in coco['images']}
    img_max = {}
    for a in coco['annotations']:
        iid = a['image_id']
        _, _, w, h = a['bbox']
        area = w * h
        if iid not in img_max or area > img_max[iid]:
            img_max[iid] = area
    keep = set()
    for iid, ma in img_max.items():
        im = img_map[iid]
        ia = im.get('width', 0) * im.get('height', 0)
        if ia > 0 and ma / ia >= threshold:
            keep.add(iid)
    return {
        'info': coco.get('info', {}), 'licenses': coco.get('licenses', []),
        'categories': coco['categories'],
        'images': [i for i in coco['images'] if i['id'] in keep],
        'annotations': [a for a in coco['annotations'] if a['image_id'] in keep]
    }


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def process_json(json_path, name):
    """筛选单JSON文件"""
    with open(json_path) as f:
        coco = json.load(f)
    filtered = filter_coco(coco)
    total, kept = len(coco['images']), len(filtered['images'])
    out = json_path.replace('.json', f'_bbox{int(THRESHOLD*100)}pct.json')
    save_json(filtered, out)
    log(f"  {name}: {total} → {kept} ({kept/total*100:.1f}%)")
    return out


# ====================== UOT100 (Kaggle) ======================

def download_uot100():
    d = os.path.join(BASE_DIR, 'UOT100')
    af = os.path.join(d, 'annotations', 'instances_train.json')
    if os.path.exists(af):
        log("UOT100 已存在"); return af
    log("下载 UOT100 (Kaggle)...")
    os.makedirs(d, exist_ok=True)
    if not shutil.which('kaggle'):
        log("安装 kaggle: pip install kaggle"); return None
    if not run_cmd(f"kaggle datasets download landrykezebou/uot100-underwater-object-tracking-dataset -p {d} --unzip"):
        return None
    return _convert_tracking(d, 'UOT100')


def _convert_tracking(data_dir, ds_name):
    """通用: 跟踪数据集→COCO"""
    ann_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    out = os.path.join(ann_dir, 'instances_train.json')
    if os.path.exists(out):
        return out
    log(f"转换 {ds_name} → COCO...")
    from PIL import Image
    images, anns, img_id, ann_id = [], [], 0, 0
    cats = [{'id': 1, 'name': 'object'}]
    for root, _, files in os.walk(data_dir):
        if 'groundtruth_rect.txt' in files:
            gt = os.path.join(root, 'groundtruth_rect.txt')
            imgs = sorted(glob.glob(os.path.join(root, '*.[jJ][pP][gG]')) +
                          glob.glob(os.path.join(root, '*.[pP][nN][gG]')))
            if not imgs:
                continue
            with open(gt) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if i >= len(imgs):
                    break
                p = line.strip().split(',')
                if len(p) < 4:
                    continue
                x, y, w, h = map(float, p[:4])
                if w <= 0 or h <= 0:
                    continue
                try:
                    pil = Image.open(imgs[i]); W, H = pil.size
                except:
                    W, H = 1920, 1080
                images.append({'id': img_id,
                    'file_name': os.path.relpath(imgs[i], data_dir),
                    'width': W, 'height': H})
                anns.append({'id': ann_id, 'image_id': img_id,
                    'category_id': 1, 'bbox': [x, y, w, h],
                    'area': w * h, 'iscrowd': 0})
                img_id += 1; ann_id += 1
    if not images:
        log(f"{ds_name}: 无数据"); return None
    save_json({'info': {'description': ds_name}, 'licenses': [],
               'categories': cats, 'images': images, 'annotations': anns}, out)
    log(f"{ds_name}: {len(images)} imgs, {len(anns)} anns")
    return out


# ====================== UVOT400 (Google Drive) ======================

GDRIVE_UVOT = '1iwba0GB4tlGLvGYiY4UmqwpqMrk4ATZz'

def download_uvot400():
    d = os.path.join(BASE_DIR, 'UVOT400')
    af = os.path.join(d, 'annotations', 'instances_train.json')
    if os.path.exists(af):
        log("UVOT400 已存在"); return af
    log("下载 UVOT400 (Google Drive)...")
    os.makedirs(d, exist_ok=True)
    if not shutil.which('gdown'):
        run_cmd("pip install gdown")
    if not run_cmd(f"gdown --folder https://drive.google.com/drive/folders/{GDRIVE_UVOT} -O {d}/images --remaining-ok"):
        log(f"手动下载: https://drive.google.com/drive/folders/{GDRIVE_UVOT} → {d}/images/")
        return None
    return _convert_tracking(d, 'UVOT400')


# ====================== USIS16K (Google Drive) ======================

GDRIVE_USIS = '14tbW3Ie8MfVjQy9DJKXnFlJ_g-6Z6xcX'

def download_usis16k():
    d = os.path.join(BASE_DIR, 'USIS16K')
    af = os.path.join(d, 'annotations', 'instances_train.json')
    if os.path.exists(af):
        log("USIS16K 已存在"); return af
    log("下载 USIS16K (Google Drive)...")
    os.makedirs(d, exist_ok=True)
    if not shutil.which('gdown'):
        run_cmd("pip install gdown")
    zp = os.path.join(d, 'USIS16K.zip')
    if not os.path.exists(zp):
        if not run_cmd(f"gdown {GDRIVE_USIS} -O {zp}"):
            log(f"手动下载: https://drive.google.com/file/d/{GDRIVE_USIS}")
            return None
    log("解压...")
    run_cmd(f"unzip -o {zp} -d {d}")
    return _convert_usis(d)


def _convert_usis(data_dir):
    ann_dir = os.path.join(data_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    out = os.path.join(ann_dir, 'instances_train.json')
    if os.path.exists(out):
        return out

    # 找已有COCO
    for jf in glob.glob(os.path.join(data_dir, '**/*.json'), recursive=True):
        try:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, dict) and 'images' in data:
                log(f"发现已有COCO: {jf}")
                shutil.copy(jf, out)
                return out
        except:
            continue

    # VOC格式
    import xml.etree.ElementTree as ET
    from PIL import Image
    images, anns = [], []
    cmap, cats = {}, []
    img_id, ann_id, cid = 0, 0, 1

    for xf in glob.glob(os.path.join(data_dir, '**/*.xml'), recursive=True):
        try:
            root = ET.parse(xf).getroot()
        except:
            continue
        fn = root.find('filename')
        if fn is None:
            continue
        fname = fn.text.strip()
        idir = os.path.dirname(xf)
        ipath = None
        for ext in ['', '.jpg', '.png', '.jpeg']:
            c = os.path.join(idir, fname + ext)
            if os.path.exists(c):
                ipath = c; break
            if fname.endswith(ext) and os.path.join(idir, fname):
                ipath = os.path.join(idir, fname); break
        if not ipath:
            continue
        try:
            pil = Image.open(ipath); W, H = pil.size
        except:
            continue
        for obj in root.findall('object'):
            ne = obj.find('name')
            if ne is None:
                continue
            cn = ne.text.strip()
            if cn not in cmap:
                cmap[cn] = cid; cats.append({'id': cid, 'name': cn}); cid += 1
            bb = obj.find('bndbox')
            if bb is None:
                continue
            bw = float(bb.find('xmax').text) - float(bb.find('xmin').text)
            bh = float(bb.find('ymax').text) - float(bb.find('ymin').text)
            if bw <= 0 or bh <= 0:
                continue
            images.append({'id': img_id,
                'file_name': os.path.relpath(ipath, data_dir),
                'width': W, 'height': H})
            anns.append({'id': ann_id, 'image_id': img_id,
                'category_id': cmap[cn],
                'bbox': [float(bb.find('xmin').text), float(bb.find('ymin').text), bw, bh],
                'area': bw * bh, 'iscrowd': 0})
            img_id += 1; ann_id += 1

    if not images:
        log("USIS16K: 无可解析标注"); return None
    save_json({'info': {'description': 'USIS16K'}, 'licenses': [],
               'categories': cats, 'images': images, 'annotations': anns}, out)
    log(f"USIS16K: {len(images)} imgs, {len(anns)} anns, {len(cats)} classes")
    return out


# ====================== CoralSCOP ======================

def download_coralscop():
    d = os.path.join(BASE_DIR, 'CoralSCOP')
    af = os.path.join(d, 'annotations', 'instances_train.json')
    if os.path.exists(af):
        log("CoralSCOP 已存在"); return af
    log("CoralSCOP 数据集 (CoralMask) 需要填写表单申请下载:")
    log("  申请链接: https://docs.google.com/forms/d/e/1FAIpQLSc8qHFBwhsJS_46hqS42NHN-3OqD5GSwvv4Sb36njdrb3LI7g/viewform")
    log("  项目主页: https://coralscop.hkustvgd.com/")
    log(f"  下载后解压到: {d}/")
    log("  要求: images/ + annotations/ 的COCO格式")
    return None


# ====================== 主函数 ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['uot100', 'uvot400', 'usis16k', 'coralscop'])
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    os.makedirs(BASE_DIR, exist_ok=True)

    handlers = {
        'uot100': (download_uot100, 'UOT100'),
        'uvot400': (download_uvot400, 'UVOT400'),
        'usis16k': (download_usis16k, 'USIS16K'),
        'coralscop': (download_coralscop, 'CoralSCOP'),
    }

    for ds in args.datasets:
        if ds not in handlers:
            log(f"未知: {ds}"); continue
        fn, name = handlers[ds]
        log(f"\n{'='*60}\n处理: {name}\n{'='*60}")
        result = fn()
        if result and isinstance(result, str) and result.endswith('.json') and not args.download_only:
            process_json(result, name)
        elif result is None:
            log(f"{name}: 跳过")

    log(f"\n完成! 数据在: {BASE_DIR}")


if __name__ == '__main__':
    main()
