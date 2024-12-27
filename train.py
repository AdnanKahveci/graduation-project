from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi
import os
import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

# GPU kullanılabilir durumda ise GPU, aksi takdirde CPU kullanılır
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argümanları komut satırından almak için argparse kullanıyoruz
parser = argparse.ArgumentParser(description="Glow trainer")
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--iter", default=200000, type=int, help="maximum iterations")
parser.add_argument(
    "--n_flow", default=32, type=int, help="number of flows in each block"
)
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument(
    "--no_lu",
    action="store_true",
    help="use plain convolution instead of LU decomposed version",
)
parser.add_argument(
    "--affine", action="store_true", help="use affine coupling instead of additive"
)
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples")
parser.add_argument("path", metavar="PATH", type=str, help="Path to image directory")

# Görüntü verilerini yükleyip ön işleme yapmak için kullanılan fonksiyon
def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    # Veriyi belirtilen dizinden yükler ve DataLoader ile iteratif bir şekilde sunar
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    # Veriyi sürekli döndürmek için iterasyon yapar
    while True:
        try:
            yield next(loader)
        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)

# z vektörlerinin şekillerini hesaplar
def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    # Her blok için giriş boyutu ve kanal sayısını küçültür
    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

# Kayıp fonksiyonunu hesaplar: log olasılık ve log determinant kullanarak
def calc_loss(log_p, logdet, image_size, n_bins):
    # Her piksel için toplam kaybı hesaplar
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )

# Eğitim fonksiyonu: Modeli eğitmek için kullanılır
def train(args, model, optimizer):
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2.0 ** args.n_bits

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    # Validation seti için
    val_dataset = iter(sample_data(args.path, args.batch, args.img_size))
    
    best_val_loss = float('inf')
    patience = 2000    # Erken durdurma için sabır
    no_improvement = 0

    # Eğitim başlangıcı için iterasyon kontrolü
    start_iter = 0

    # Eğer bir checkpoint varsa modeli ve optimizer'ı o noktadan yükleyin
    checkpoint_path = "checkpoint/checkpoint_latest.pth"  # Son kaydedilen checkpoint'i yükleme
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iteration']  # Eğitime kaldığınız iterasyondan devam
        best_val_loss = checkpoint['best_val_loss']
        no_improvement = checkpoint['no_improvement']
        print(f"Checkpoint yükleniyor, iterasyon: {start_iter}")
    else:
        print("Checkpoint bulunamadı, sıfırdan başlanıyor.")

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=1e-6)
    validation_losses = []

    with tqdm(range(start_iter, args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)
            image = image * 255

            if args.n_bits < 8:
                image = torch.floor(image / 2 ** (8 - args.n_bits))

            image = image / n_bins - 0.5

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(image + torch.rand_like(image) / n_bins)
                    continue
            else:
                log_p, logdet, _ = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet, args.img_size, n_bins)
            model.zero_grad()
            loss.backward()

            optimizer.step()

            # Öğrenme oranı planlaması
            scheduler.step()

            warmup_lr = scheduler.get_last_lr()[0]
            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )

            # Model ve optimizer durumunu kaydetme (daha sık checkpoint)
            if i % 100 == 0:
                torch.save({
                    'iteration': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'no_improvement': no_improvement
                }, f"checkpoint/checkpoint_latest.pth")

            if i % 100 == 0:
                with torch.no_grad():
                    utils.save_image(
                        model_single.reverse(z_sample).cpu().data,
                        f"sample/{str(i + 1).zfill(6)}.png",
                        normalize=True,
                        nrow=10,
                        value_range=(-0.5, 0.5),
                    )
            # Validation ve erken durdurma
            if i % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    val_image, _ = next(val_dataset)
                    val_image = val_image.to(device)
                    val_image = val_image * 255
                    if args.n_bits < 8:
                        val_image = torch.floor(val_image / 2 ** (8 - args.n_bits))
                    val_image = val_image / n_bins - 0.5
                    val_log_p, val_logdet, _ = model(val_image + torch.rand_like(val_image) / n_bins)
                    val_loss, _, _ = calc_loss(val_log_p, val_logdet, args.img_size, n_bins)

                    validation_losses.append(val_loss.item())

                    print(f"Validation Loss at Iteration {i}: {val_loss.item():.5f}")

                    plt.plot(validation_losses, label='Validation Loss')
                    plt.xlabel('Validation Checkpoints')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig(f'training_plots/validation_loss_plot_{i}.png')  # İterasyona göre farklı dosya adı
                    plt.close()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement = 0
                    torch.save(model.state_dict(), "checkpoint/best_model.pt")
                else:
                    no_improvement += 1
                
                if no_improvement >= patience:
                    print(f"Early stopping at iteration {i}")
                    break
                
                model.train()

            # Düzenli değerlendirme
            if i % 5000 == 0:
                print(f"\nIteration {i}: Detailed Evaluation")
                print(f"Training Loss: {loss.item():.5f}")
                print(f"Validation Loss: {val_loss.item():.5f}")
                print(f"Best Validation Loss: {best_val_loss:.5f}")
                print(f"Learning Rate: {warmup_lr:.7f}")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    model = nn.DataParallel(model_single)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)
