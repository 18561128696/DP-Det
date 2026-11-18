import os
from pathlib import Path
from PIL import Image
import shutil

# --- 1. 路径配置 (根据你的项目结构自动定位) ---
# 脚本在 scripts/ 目录下，所以 .parent.parent 回到根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 原始数据位置 (你解压的地方)
# 假设你在 datasets/VisDrone_Raw 下解压了那三个文件夹
RAW_DIR = BASE_DIR / 'datasets' / 'VisDrone_Raw'

# 输出数据位置 (YOLO 需要的格式)
OUT_DIR = BASE_DIR / 'datasets' / 'VisDrone_YOLO'


def convert_box(size, box):
    # VisDrone(x,y,w,h) -> YOLO(center_x, center_y, w, h) + 归一化
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


def process_dataset(split_name):
    from tqdm import tqdm
    # 原始路径
    vis_img_dir = RAW_DIR / split_name / 'images'
    vis_lbl_dir = RAW_DIR / split_name / 'annotations'

    # 目标路径
    save_img_dir = OUT_DIR / split_name / 'images'
    save_lbl_dir = OUT_DIR / split_name / 'labels'

    # 检查原始数据是否存在
    if not vis_img_dir.exists():
        print(f"⚠️ 警告: 找不到原始数据目录: {vis_img_dir}")
        print(f"   请确保你把解压后的文件夹放在了 datasets/VisDrone_Raw/ 下")
        return

    # 创建输出目录
    save_img_dir.mkdir(parents=True, exist_ok=True)
    save_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 正在转换: {split_name} ...")

    # 扫描所有图片
    image_files = list(vis_img_dir.glob('*.jpg'))

    for img_path in tqdm(image_files):
        # 1. 复制图片 (复制一份到新目录，保证原始数据纯净)
        # 如果你想节省硬盘空间，可以把 shutil.copy 改成 os.link (创建硬链接)
        if not (save_img_dir / img_path.name).exists():
            shutil.copy(img_path, save_img_dir / img_path.name)

        # 2. 处理标签
        lbl_name = img_path.stem + '.txt'
        src_lbl_path = vis_lbl_dir / lbl_name
        dst_lbl_path = save_lbl_dir / lbl_name

        # 获取图片尺寸用于归一化
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except:
            continue

        # 如果没有对应的标签文件(可能是纯背景图)，生成空文件
        if not src_lbl_path.exists():
            with open(dst_lbl_path, 'w') as f: pass
            continue

        with open(src_lbl_path, 'r') as f:
            lines = f.readlines()

        yolo_lines = []
        for line in lines:
            data = line.strip().split(',')
            if len(data) < 8: continue

            # VisDrone 类别: 0(ignored), 1(pedestrian), ... 11(others)
            category = int(data[5])

            # 过滤掉不需要的类别
            if category == 0 or category == 11: continue

            # 类别 ID 修正: VisDrone 1-10 -> YOLO 0-9
            cls_id = category - 1

            box = (float(data[0]), float(data[1]), float(data[2]), float(data[3]))

            # 坐标转换
            bb = convert_box((w, h), box)

            # 边界检查 (防止标注超出图片)
            bb = [max(0, min(1, x)) for x in bb]

            yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

        with open(dst_lbl_path, 'w') as f:
            f.writelines(yolo_lines)


def main():
    import argparse
    global RAW_DIR, OUT_DIR
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO; categories 0/11 are excluded.")
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=BASE_DIR / 'datasets/VisDrone_YOLO')
    args = parser.parse_args()
    RAW_DIR, OUT_DIR = args.raw_root.resolve(), args.output.resolve()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise SystemExit('Output is not empty; choose a new directory to preserve existing data.')
    for split in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        process_dataset(split)

if __name__ == '__main__':
    main()
