import argparse
import torch
from torchvision import utils
from model import Glow  # Glow modelini içeren modül

# GPU kullanılabilir durumda ise GPU, aksi takdirde CPU kullanılır
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argümanları komut satırından almak için argparse kullanıyoruz
parser = argparse.ArgumentParser(description="Glow Image Generator")
parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--no_lu", action="store_true", help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")
parser.add_argument("--img_size", default=64, type=int, help="image size")
parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
parser.add_argument("--n_sample", default=20, type=int, help="number of samples to generate")
parser.add_argument("--model_path", type=str, default="checkpoint/best_model.pt", help="path to the trained model checkpoint")
parser.add_argument("--output_dir", type=str, default="sample", help="directory to save generated images")

# Görüntü üretimi için z vektörlerinin şekillerini hesaplayan fonksiyon
def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []
    for _ in range(n_block - 1):
        input_size //= 2
        n_channel *= 2
        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))
    return z_shapes

# Görüntü üretimi fonksiyonu
def generate_images(args):
    # Modeli yükleme
    model_single = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu)
    model = torch.nn.DataParallel(model_single)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Z vektörlerini oluşturma
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    z_sample = [torch.randn(args.n_sample, *z).to(device) * args.temp for z in z_shapes]

    # Görüntüleri üretme
    with torch.no_grad():
        generated_images = model_single.reverse(z_sample).cpu()
        utils.save_image(
            generated_images,
            f"{args.output_dir}/sample.png",
            normalize=True,
            nrow=10,
            value_range=(-0.5, 0.5),
        )
    print(f"Görüntüler {args.output_dir} dizinine kaydedildi.")

if __name__ == "__main__":
    args = parser.parse_args()
    generate_images(args)
# Bu kod bloğu, Glow modelini kullanarak belirtilen sayıda örnek görüntü üretir ve kaydeder.