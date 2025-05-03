import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import tifffile
from skimage.metrics import structural_similarity as ssim_func
from skimage.transform import resize
from config import config                                  # 配置文件路径 & 参数
from model.F_VCD import F_Denoise, F_Recon
from tensorlayer.layers import InputLayer
from utils import normalize_percentile
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.TRAIN.device)

def read_valid_npy_images(path):
    img_list = sorted([f for f in os.listdir(path) if f.endswith('.npy')])
    if not img_list:
        raise FileNotFoundError(f"❌ 没有找到任何 .npy 验证数据: {path}")
    imgs = []
    for fn in img_list:
        data = np.load(os.path.join(path, fn)).astype(np.float32)
        if data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        imgs.append(normalize_percentile(data))  # 使用与训练时一致的 normalize_percentile 归一化方法
    imgs = np.stack(imgs, axis=0)  # [N, H, W, V]
    print(f"✅ 加载 {len(img_list)} 张验证图, shape = {imgs.shape}")
    return imgs, img_list

def calculate_psnr(gt, pred, data_range=1.0):
    """计算单张 2D 切片的 PSNR"""
    mse = np.mean((gt - pred) ** 2)
    return 100.0 if mse == 0 else 20 * np.log10(data_range / np.sqrt(mse))

def calculate_ssim(gt, pred, data_range=1.0):
    """调用 skimage.metrics 计算 SSIM"""
    return ssim_func(gt, pred, data_range=data_range)

def infer(epoch=0, batch_size=1):
    """主流程：构图→载权重→批量推理→计算指标并保存"""
    epoch_tag = 'best' if epoch == 0 else f'epoch{epoch}'
    ckpt_dir   = config.TRAIN.ckpt_dir
    valid_path = config.VALID.lf2d_path               # 验证集 LFP 路径 :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}
    save_dir   = config.VALID.saving_path
    os.makedirs(save_dir, exist_ok=True)

    # 1) 读取验证集
    valid_imgs, names = read_valid_npy_images(valid_path)
    H, W = valid_imgs.shape[1], valid_imgs.shape[2]
    SR_size    = np.array([H, W]) * config.img_setting.sr_factor
    Recon_size = SR_size * np.array(config.img_setting.ReScale_factor)

    # 2) 构建图
    tf.reset_default_graph()
    t_image = tf.placeholder(tf.float32, [batch_size, H, W, config.img_setting.Nnum], name='t_LFP')
    with tf.device(f"/gpu:{config.TRAIN.device}"):
        inp_layer = InputLayer(t_image, name='input_LF')
        SR_net = F_Denoise(
            inp_layer, output_size=SR_size,
            angRes=config.img_setting.Nnum,
            sr_factor=config.img_setting.sr_factor,
            reuse=False, name=config.net_setting.SR_model,
            channels_interp=config.channels_interp,
            normalize_mode=config.preprocess.normalize_mode
        )
        Recon_net = F_Recon(
            SR_net.outputs,    # ← 只传 outputs 张量
            n_slices=config.img_setting.n_slices,
            output_size=Recon_size,
            is_train=False,
            reuse=False,
            name=config.net_setting.Recon_model,
            channels_interp=config.channels_interp,
            normalize_mode=config.preprocess.normalize_mode
        )

    # 3) 查找权重文件
    def find_ckpt(model_name):
        cand = [f for f in os.listdir(ckpt_dir)
                if f.endswith('.npz') and epoch_tag in f and model_name in f]
        if not cand:
            alias = 'SR_net' if 'F_Denoise' in model_name else 'recon_net'
            cand = [f for f in os.listdir(ckpt_dir)
                    if f.endswith('.npz') and epoch_tag in f and alias in f]
        if not cand:
            raise FileNotFoundError(f"❌ 找不到 {model_name} 的权重文件 ({epoch_tag})")
        return os.path.join(ckpt_dir, cand[0])

    sr_ckpt    = find_ckpt(config.net_setting.SR_model)
    recon_ckpt = find_ckpt(config.net_setting.Recon_model)
    print("📂 SR 权重路径:", sr_ckpt)
    print("📂 Recon 权重路径:", recon_ckpt)

    # 4) 会话与载权
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())
    # 【2】在这里打印部分 Recon 权重的均值、方差
    for v in Recon_net.all_params[:5]:
        arr = sess.run(v)
        print(f"[WEIGHT STAT] {v.name:30s} mean={arr.mean():.6f}, std={arr.std():.6f}")
    
    # —— 手动加载 SR_net 权重 —— 
    sr_npz = np.load(sr_ckpt, allow_pickle=True)
    matched_sr = 0
    for var in SR_net.all_params:
        key = var.name  # 包含 ":0"
        if key in sr_npz:
            sess.run(var.assign(sr_npz[key]))
            matched_sr += 1
        else:
            print(f"⚠️ SR_net: 未在 npz 中找到变量 {key}")
    print(f"✅ SR_net 成功加载 {matched_sr}/{len(SR_net.all_params)} 个参数")

    # —— 手动加载 Recon_net 权重 —— 
    print(f"▶ Loading Recon weights from {recon_ckpt}")
    recon_npz = np.load(recon_ckpt, allow_pickle=True)
    matched_recon = 0
    for var in Recon_net.all_params:
        key = var.name
        if key in recon_npz:
            sess.run(var.assign(recon_npz[key]))
            matched_recon += 1
        else:
            print(f"⚠️ Recon_net: 未在 npz 中找到变量 {key}")
    print(f"✅ Recon_net 成功加载 {matched_recon}/{len(Recon_net.all_params)} 个参数")
    
    # 【1.1】只跑 SR 阶段，查看输出 —— 一定要用 valid_imgs 而不是 fake_sr
    sr_out = sess.run(SR_net.outputs, feed_dict={t_image: valid_imgs[0:1]})
    print(f"[SR TEST] SR_out min={sr_out.min():.4f}, max={sr_out.max():.4f}")

    # 【1.2】只跑 Recon 阶段用“理想超分”作为输入
    # 读取GT图像并归一化
    gt_vol = tifffile.imread(os.path.join(config.VALID.gt_path, names[0].replace('.npy', '.tif')))
    print(f"[DEBUG] gt_vol min={gt_vol.min()}, max={gt_vol.max()}")  # 归一化前
    gt_vol = gt_vol.astype(np.float32) / gt_vol.max()  # 归一化处理
    print(f"[DEBUG] gt_vol after normalization min={gt_vol.min()}, max={gt_vol.max()}")  # 归一化后

    # 打印每个切片的最小值和最大值，确保数据没有问题
    for i in range(gt_vol.shape[0]):
        print(f"[DEBUG] GT_vol slice {i} min={gt_vol[i].min()}, max={gt_vol[i].max()}")
    
    # 确保 gt_vol 里面的有效数据
    valid_slices = [i for i in range(gt_vol.shape[0]) if np.any(gt_vol[i] != 0)]
    print(f"[DEBUG] Valid slices: {valid_slices}")
    
    # 确保 gt_vol 有有效切片
    if valid_slices:
        mid = valid_slices[len(valid_slices) // 2]  # 使用有效切片的中间切片
        fake_sr = gt_vol[mid, ...][None, ..., None]  # shape [1, H, W, 1]
        fake_sr = np.repeat(fake_sr, valid_imgs.shape[-1], axis=-1)  # shape [1, H, W, Nnum]
    else:
        print("[ERROR] No valid slices found.")
        fake_sr = np.zeros((1, 160, 160, 3))  # 处理没有有效切片的情况

    # 使用 normalize_percentile 归一化方法，确保与训练时一致
    fake_sr = normalize_percentile(fake_sr)
    print(f"[DEBUG] fake_sr min={fake_sr.min()}, max={fake_sr.max()}, mean={fake_sr.mean()}")

    # 调整 fake_sr 的形状为 [1, 160, 160, 3]（通过保持批量维度）
    fake_sr_resized = cv2.resize(fake_sr[0], (160, 160), interpolation=cv2.INTER_LINEAR)

    # 打印调整后的图像范围
    print(f"[DEBUG] fake_sr_resized min={fake_sr_resized.min()}, max={fake_sr_resized.max()}")

    # 确保调整后的图像形状为 [1, 160, 160, Nnum]，即增加批量维度
    fake_sr_resized = fake_sr_resized[None, ...]  # 增加一个新的维度，使形状变为 [1, 160, 160, Nnum]

    # 再次打印 fake_sr_resized 的范围
    print(f"[DEBUG] fake_sr_resized min={fake_sr_resized.min()}, max={fake_sr_resized.max()}")

    # 打印 fake_sr 的形状，确保它符合网络的输入要求
    print(f"[DEBUG] fake_sr shape before SR_net: {fake_sr_resized.shape}")

    # 跑 SR_net 得到 sr_out: [1, H*sr_factor, W*sr_factor, Nnum]
    sr_out = sess.run(SR_net.outputs, feed_dict={t_image: fake_sr_resized})
    print(f"[DEBUG] sr_out min={sr_out.min()}, max={sr_out.max()}, mean={sr_out.mean()}")

    # 保存 SR 输出的图像
    save_sr_path = os.path.join(save_dir, 'sr_out.png')
    tifffile.imwrite(save_sr_path, sr_out[0])

    # 打印 sr_out 的形状，确保它符合预期
    print(f"[DEBUG] sr_out shape after SR_net: {sr_out.shape}")

    # 打印 sr_out 的最小值和最大值，确保其值范围合理
    print(f"[DEBUG] sr_out min={sr_out.min()}, max={sr_out.max()}")

    # resize 回 [1,160,160,Nnum]
    H0, W0 = valid_imgs.shape[1], valid_imgs.shape[2]  # 160,160
    sr_down = cv2.resize(
        sr_out[0], 
        (W0, H0),  # OpenCV expects width, height (W, H) instead of (H, W)
        interpolation=cv2.INTER_LINEAR  # 使用线性插值
    )

    # 确保调整后的图像形状为 [1, H0, W0, Nnum]，其中 Nnum 是原始图像的通道数
    sr_down = sr_down[None, ...]  # 增加一个新的维度，使形状变为 [1, H0, W0, Nnum]

    # 再跑 Recon_net
    recon_out = sess.run(Recon_net.outputs, feed_dict={t_image: sr_down})
    print(f"[RECON TEST] Recon_out min={recon_out.min():.4f}, max={recon_out.max():.4f}")

    # 保存 Recon 输出的图像
    save_recon_path = os.path.join(save_dir, 'recon_out.png')
    tifffile.imwrite(save_recon_path, recon_out[0])

    print(f"📂 保存 SR 输出至 {save_sr_path}, Recon 输出至 {save_recon_path}")

    # 5) 批量推理、保存 tif、计算并记录 PSNR/SSIM
    log_path = os.path.join(save_dir, 'validation_metrics.txt')
    with open(log_path, 'w') as logf:
        for idx in range(len(valid_imgs)):
            batch = valid_imgs[idx:idx+batch_size]
            out = sess.run(Recon_net.outputs, feed_dict={t_image: batch})
            vol = np.squeeze(out, axis=0)
            vol = np.clip(vol, 0, 1)

            # 保存重建体数据
            save_p = os.path.join(save_dir, names[idx].replace('.npy', '.tif'))
            tifffile.imwrite(save_p, vol.astype(np.float32))

            # 读取 GT stack 并归一化
            gt_p = os.path.join(config.VALID.gt_path, names[idx].replace('.npy', '.tif'))
            if os.path.exists(gt_p):
                gt_vol = tifffile.imread(gt_p).astype(np.float32)
                gt_vol = gt_vol / np.max(gt_vol)
                # —— 新增：如果 gt_vol 是 (D,H,W)，就转成 (H,W,D)，否则保持不变 —— 
                if gt_vol.ndim == 3 and gt_vol.shape[0] == vol.shape[-1]:
                    gt_vol = np.transpose(gt_vol, (1,2,0))
                    print(f"[DEBUG] Transposed gt_vol to shape {gt_vol.shape}")

                psnrs, ssims = [], []
                for z in range(vol.shape[-1]):
                    gt_slice   = gt_vol[..., z]    # 现在一定是 (H, W)
                    pred_slice = vol[...,   z]     # (H, W)
                    # 跳过纯背景切片
                    if gt_slice.max() < 1e-6:
                        continue
                    psnrs.append(calculate_psnr( gt_slice, pred_slice, data_range=1.0))
                    ssims.append(calculate_ssim(gt_slice, pred_slice, data_range=1.0))
                avg_psnr = float(np.mean(psnrs)) if psnrs else float('nan')
                avg_ssim = float(np.mean(ssims)) if ssims else float('nan')
            else:
                avg_psnr = avg_ssim = float('nan')

            logf.write(f"{names[idx]}  PSNR:{avg_psnr:.4f}  SSIM:{avg_ssim:.4f}\n")
            print(f"\r[{idx+1}/{len(valid_imgs)}] PSNR:{avg_psnr:.2f} dB, SSIM:{avg_ssim:.4f}", end='')

    print(f"\n✅ 推理完成，日志保存在：{log_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='F-VCD 推理与评估脚本')
    parser.add_argument('-c', '--ckpt',  type=int, default=0, help='0 表示 best，否则表示 epoch 编号')
    parser.add_argument('-b', '--batch', type=int, default=1, help='推理时 batch size')
    args = parser.parse_args()
    infer(epoch=args.ckpt, batch_size=args.batch)
