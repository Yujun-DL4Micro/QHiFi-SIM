import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
from models.QHiFi import QHiFi
from util.util import img_comp
from loss.SSIM_loss import SSIM
import matplotlib.pyplot as plt
import random
import logging
import time


def train(args, train_loader, eval_loader, device):
    eval_dataset = eval_loader.dataset
    # Initialize model
    if args.model_name == 'QHiFi':
        print("Training model: QHiFi")

        qd_model = QHiFi(
            disable_ssfe=args.disable_ssfe,
            disable_wife=args.disable_wife,
            disable_channel_attn=args.disable_channel_attn,
            disable_spatial_attn=args.disable_spatial_attn,
            disable_identity=args.disable_identity
        ).to(device)

    else:
        raise ValueError(f"Error: the model is not implemented!!!!")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{args.experiment_name or 'exp'}_{timestamp}"
    log_file = f"train_{experiment_id}.log"

    optimizer = optim.Adam(qd_model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=f'runs/{experiment_id}')
    L1_loss = nn.L1Loss()
    ssim_loss = SSIM()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    start_epoch = 0
    best_eval_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, weights_only=True)
        qd_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
        start_epoch = checkpoint['epoch']
        best_eval_loss = checkpoint.get('best_eval_loss', best_eval_loss)
        logger.info(f"Resumed from checkpoint: epoch {start_epoch}, best eval loss {best_eval_loss:.4f}")

    print("start training>>>>>>>>>")
    for epoch in range(start_epoch, args.nEpochs):
        qd_model.train()
        running_loss = 0.0
        current_eval_loss = None
        for i, (wf_images, hr_images, psf_tensors) in enumerate(train_loader):
            qd_model.train()
            wf_image = wf_images.to(device)
            hr_image = hr_images.to(device)
            psf_tensor = psf_tensors.to(device)

            corrected_image, degraded_corrected, dg_sr = qd_model(hr_image, wf_image, psf_tensor)

            loss1 = args.weight * L1_loss(degraded_corrected, wf_image)
            if not args.disable_ssim_loss:
                loss1 += (1 - args.weight) * torch.abs(1 - ssim_loss(degraded_corrected, wf_image))

            if args.disable_loss2:
                loss = loss1
            else:
                loss2 = args.loss_corrected_l1 * L1_loss(corrected_image, hr_image)
                if not args.disable_ssim_loss:
                    loss2 += args.loss_corrected_ssim * torch.abs(1 - ssim_loss(corrected_image, hr_image))
                loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # ------------------------------------------- evaluate -------------------------------------------------
            if (i + 1) % args.validate_interval == 0:
                with torch.no_grad():
                    num_batches = len(eval_dataset)
                    random_indices = random.sample(range(num_batches), args.validate_num)
                    random_indices.sort()
                    sampler = SubsetRandomSampler(random_indices)
                    subset_eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=sampler)
                    qd_model.eval()

                    psnr_val, ssim_val, mse_val, nrmses_val, eval_loss = evaluation(
                        args, subset_eval_loader, qd_model, device, L1_loss, ssim_loss
                    )

                    current_eval_loss = eval_loss
                    writer.add_scalar('psnr_val', psnr_val, epoch * len(train_loader) + i)
                    writer.add_scalar('ssim_val', ssim_val, epoch * len(train_loader) + i)
                    writer.add_scalar('mse_val', mse_val, epoch * len(train_loader) + i)
                    writer.add_scalar('nrmses_val', nrmses_val, epoch * len(train_loader) + i)
                    writer.add_scalar('eval_loss', eval_loss, epoch * len(train_loader) + i)
                writer.add_scalar('train_loss', running_loss / args.validate_interval, epoch * len(train_loader) + i)

                print(f'Epoch [{epoch + 1}/{args.nEpochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Eval Loss: {eval_loss:.4f},'
                      f'train Loss: {running_loss / args.validate_interval:.4f},'
                      f'Eval_PSNR: {psnr_val:.4f}, Eval_SSIM: {ssim_val:.4f}, Eval_mse: {mse_val:.4f}, Eval_nrmses: {nrmses_val:.4f}')

                logger.info(
                    f"Epoch: {epoch}, PSNR: {psnr_val:.3f}, SSIM: {ssim_val:.6f}, MSE: {mse_val:.6f}, NRMSES: {nrmses_val:.6f}"
                )
                running_loss = 0.0

            if (i + 1) % args.sample_interval == 0:
                iter_v = epoch * len(train_loader) + i
                with torch.no_grad():
                    num_batches = len(eval_dataset)
                    random_indices = random.sample(range(num_batches), args.sample_num)
                    random_indices.sort()
                    sampler = SubsetRandomSampler(random_indices)
                    subset_eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=sampler)
                    qd_model.eval()
                    output_path = args.output_path
                    sample_1(iter_v, subset_eval_loader, qd_model, output_path, device)

        if running_loss != 0.0:
            writer.add_scalar('train_loss', running_loss / (len(train_loader) % args.validate_interval),
                              epoch * len(train_loader) + i)
            running_loss = 0.0

        if current_eval_loss is not None:
            scheduler.step(current_eval_loss)
        if (epoch + 1) % args.model_save_frequency == 0:
            os.makedirs(args.checkpoint_path, exist_ok=True)
            filename = 'checkpoint' + '_epoch' + str(epoch + 1) + '.pth.tar'
            save_path = os.path.join(args.checkpoint_path, filename)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': qd_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_loss': current_eval_loss,
                'best_eval_loss': best_eval_loss,
            }, save_path)
            print(f'Saved checkpoint for epoch {epoch + 1}')

        if current_eval_loss is not None and current_eval_loss < best_eval_loss:
            best_eval_loss = current_eval_loss
            best_save_path = os.path.join(args.checkpoint_path, 'best_checkpoint.pth.tar')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': qd_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'eval_loss': best_eval_loss,
            }, best_save_path)
            print(f'Updated best checkpoint (eval loss: {best_eval_loss:.4f})')
    writer.close()


def evaluation(args, subset_eval_loader, qd_model, device, L1_loss, ssim_loss):
    qd_model.eval()
    eval_loss = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum = 0.0
    nrmses_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for eval_idx, (wf_images, hr_images, psf_tensors) in enumerate(subset_eval_loader):
            val_wf = wf_images.to(device)
            val_hr = hr_images.to(device)
            val_psf_tensor = psf_tensors.to(device)

            val_corrected_image, val_final_output, val_dg_sr = qd_model(val_hr, val_wf, val_psf_tensor)

            val_loss1 = args.weight * L1_loss(val_final_output, val_wf)
            if not args.disable_ssim_loss:
                val_loss1 += (1 - args.weight) * torch.abs(1 - ssim_loss(val_final_output, val_wf))

            if args.disable_loss2:
                val_loss = val_loss1
            else:
                val_loss2 = args.loss_corrected_l1 * L1_loss(val_corrected_image, val_hr)
                if not args.disable_ssim_loss:
                    val_loss2 += args.loss_corrected_ssim * torch.abs(1 - ssim_loss(val_corrected_image, val_hr))
                val_loss = val_loss1 + val_loss2

            eval_loss += val_loss.item() * val_wf.size(0)

            val_degraded_corrected = val_final_output.cpu().numpy().transpose(0, 2, 3, 1)
            val_wf_np = val_wf.cpu().numpy().transpose(0, 2, 3, 1)

            batch_size = val_wf_np.shape[0]
            total_samples += batch_size
            for pr, gt in zip(val_degraded_corrected, val_wf_np):
                psnrs, ssims, mses, nrmses = img_comp(gt, pr, data_range=1.0)
                psnr_sum += psnrs
                ssim_sum += ssims
                mse_sum += mses
                nrmses_sum += nrmses

    psnr_val = psnr_sum / total_samples
    ssim_val = ssim_sum / total_samples
    mse_val = mse_sum / total_samples
    nrmses_val = nrmses_sum / total_samples
    eval_loss = eval_loss / total_samples

    return psnr_val, ssim_val, mse_val, nrmses_val, eval_loss


def sample_1(iter_v, subset_eval_loader, qd_model, output_path, device):
    qd_model.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    r, c = 3, 5
    psnrs, ssims, mses, nrmses, = [], [], [], []
    ori_wf_show, ori_sr_show, out_sr_show, out_wf_show, out_srd_show = [], [], [], [], []

    for eval_idx, (wf_images, hr_images, psf_tensors) in enumerate(subset_eval_loader):
        val_wf = wf_images.to(device)
        val_hr = hr_images.to(device)
        val_psf_tensor = psf_tensors.to(device)

        val_corrected_image, val_final_output, val_dg_sr = qd_model(val_hr, val_wf, val_psf_tensor)

        val_degraded_corrected = val_final_output.cpu().numpy().transpose(0, 2, 3, 1)
        val_corrected_image = val_corrected_image.cpu().numpy().transpose(0, 2, 3, 1)
        val_wf = val_wf.cpu().numpy().transpose(0, 2, 3, 1)
        val_hr = val_hr.cpu().numpy().transpose(0, 2, 3, 1)
        val_degraded_hr = val_dg_sr.cpu().numpy().transpose(0, 2, 3, 1)

        for ori_wf, ori_sr, out_sr, out_wf, out_dg_sr in zip(val_wf, val_hr, val_corrected_image,
                                                                  val_degraded_corrected, val_degraded_hr):
            psnr, ssim, mse, nrmse = img_comp(ori_wf, out_wf, data_range=1.0)
            ori_wf_show.append(np.squeeze(ori_wf))
            ori_sr_show.append(np.squeeze(ori_sr))
            out_sr_show.append(out_sr)
            out_wf_show.append(out_wf)
            out_srd_show.append(out_dg_sr)
            psnrs.append(psnr)
            ssims.append(ssim)
            mses.append(mse)
            nrmses.append(nrmse)
    psnrs = np.array(psnrs)
    ssims = np.array(ssims)
    mses = np.array(mses)
    # show some examples
    fig, axs = plt.subplots(r, c, figsize=(20, 4 * r))

    cnt = 0
    for row in range(r):
        axs[row, 1].set_title('MSE=%.4f, SSIM=%.4f, PSNR=%.4f' % (mses[row], ssims[row], psnrs[row]), loc='center')

        for col, image in enumerate([ori_wf_show, out_wf_show, ori_sr_show, out_sr_show, out_srd_show]):
            axs[row, col].imshow(np.squeeze(image[row]))
            axs[row, col].axis('off')
            axs[row, col].set_aspect('equal')
        cnt += 1

    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    plt.tight_layout(pad=2.0)
    output_file = os.path.join(output_path, f'{iter_v}.png')
    fig.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    return
