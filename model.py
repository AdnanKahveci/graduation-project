import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la

# logabs: Mutlak değerinin logaritmasını hesaplayan fonksiyon
logabs = lambda x: torch.log(torch.abs(x))

# ActNorm sınıfı, kanal bazlı normalizasyon işlemi yapar
class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        # Konum ve ölçek parametreleri öğrenilebilir
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        # Parametrelerin ilk kez başlatılıp başlatılmadığını kontrol eder
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    # İlk başlatma işlemi, girişin ortalama ve standart sapmasını hesaplar
    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)  # Ortalamayı sıfır olacak şekilde kaydırma
            self.scale.data.copy_(1 / (std + 1e-6))  # Standart sapmayı ölçeklendirme

    # İleri yönlü geçiş işlemi, giriş verisini normalize eder
    def forward(self, input):
        _, _, height, width = input.shape

        # Eğer başlatılmamışsa, başlatmayı yap
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        # logdet (determinantın logu), modelde log olasılık hesabı için kullanılır
        logdet = height * width * torch.sum(log_abs)

        # İleri geçiş: normalize edilmiş çıktı ve log determinantı döndürülür
        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    # Ters yönlü geçiş işlemi, normalizasyonun tersini yapar
    def reverse(self, output):
        return output / self.scale - self.loc

# InvConv2d sınıfı, ters çevrilebilir 2D konvolüsyon katmanı
class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # Rastgele bir ağırlık matrisi oluşturulur ve ortogonal hale getirilir (QR ayrıştırması)
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    # İleri geçiş işlemi: Konvolüsyon ve log determinant hesabı
    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    # Ters geçiş işlemi: Konvolüsyonun tersini uygular
    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

# InvConv2dLU sınıfı, LU ayrıştırması ile ters çevrilebilir 2D konvolüsyon katmanı
class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        # LU ayrıştırması kullanarak ağırlık matrisini üçe böler (W_p, W_l, W_u)
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        # Ağırlıklar ve maskeler Torch tensörlerine çevrilir
        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    # İleri geçiş işlemi: LU konvolüsyonu ve log determinant hesabı
    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    # Ağırlık matrisini hesaplar (LU ayrıştırması kullanarak)
    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    # Ters geçiş işlemi: Konvolüsyonun tersini uygular
    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

# ZeroConv2d sınıfı, sıfırdan başlatılan 2D konvolüsyon katmanı
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    # İleri geçiş işlemi: Girdiyi pad'ler ve konvolüsyon uygular
    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out

# AffineCoupling sınıfı, ayrık affine coupling katmanı (Normalizasyon için kullanılır)
class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        # Ağ, çift girişli bir yapıyı işler ve sıfırdan başlatılmış konvolüsyon uygular
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    # İleri geçiş işlemi: Affine coupling uygular ve log determinant hesaplar
    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    # Ters geçiş işlemi: Affine coupling'in tersini uygular
    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

# Flow sınıfı, tersinir bir katman yapısını temsil eder. 
# ActNorm, InvConv2dLU ve AffineCoupling gibi katmanları birleştirir.
class Flow(nn.Module):
    def __init__(self, in_channel, affine=True):
        super().__init__()

        # Sırasıyla ActNorm, InvConv2dLU ve AffineCoupling katmanlarını oluşturur
        self.actnorm = ActNorm(in_channel)
        self.invconv = InvConv2dLU(in_channel)
        self.coupling = AffineCoupling(in_channel, affine=affine)

    # İleri geçiş işlemi: Girdiyi sırayla katmanlardan geçirir ve toplam log determinantı hesaplar
    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1 + (det2 if det2 is not None else 0)
        return out, logdet

    # Ters geçiş işlemi: Çıktıyı ters sırayla katmanlardan geçirerek girişe ulaşır
    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input

# Block sınıfı, bir dizi Flow katmanını birleştirerek modelin daha derin olmasını sağlar.
class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True):
        super().__init__()

        # Her Block belirli sayıda Flow katmanı içerir
        self.flows = nn.ModuleList()
        for _ in range(n_flow):
            self.flows.append(Flow(in_channel, affine=affine))

        self.split = split

        # Eğer split işlemi aktifse, Squeeze işlemi ile giriş boyutunu yarıya indirir
        if split:
            self.prior = ZeroConv2d(in_channel // 2, in_channel)
        else:
            self.prior = ZeroConv2d(in_channel, in_channel * 2)

    # İleri geçiş işlemi: Girdiyi Flow katmanlarından geçirir ve gerekiyorsa boyutunu yarıya indirir
    def forward(self, input):
        logdet = 0

        out = input
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_likelihood(mean, log_sd, z_new)
        else:
            z_new = out
            log_p = 0

        return out, logdet, log_p, z_new

    # Ters geçiş işlemi: Çıktıyı ters sırayla Flow katmanlarından geçirir
    def reverse(self, output, z=None, reconstruct=False):
        if reconstruct:
            input = torch.cat([output, z], 1)
        else:
            if self.split:
                mean, log_sd = self.prior(output).chunk(2, 1)
                z = gaussian_sample(mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                input = output

        for flow in reversed(self.flows):
            input = flow.reverse(input)

        return input

# gaussian_likelihood fonksiyonu, Gauss olasılık hesaplaması yapar
def gaussian_likelihood(mean, log_sd, x):
    log_p = -0.5 * log(2 * pi) - log_sd - 0.5 * ((x - mean) ** 2 / torch.exp(log_sd * 2))
    return log_p.sum([1, 2, 3])

# gaussian_sample fonksiyonu, Gauss örneklemesi yapar
def gaussian_sample(mean, log_sd):
    return mean + torch.exp(log_sd) * torch.randn_like(mean)

# Glow modeli, birçok Block ve Squeeze katmanından oluşan ana modeldir.
class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True):
        super().__init__()

        # Glow modeli, belirli sayıda Block içerir
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for _ in range(n_block):
            self.blocks.append(Block(n_channel, n_flow, split=True, affine=affine))
            n_channel *= 2

    # İleri geçiş işlemi: Girdiyi katmanlardan geçirerek log determinant ve log olasılığı hesaplar
    def forward(self, input):
        log_p = 0
        logdet = 0
        out = input

        z_outs = []

        for block in self.blocks:
            out, det, log_p_, z_new = block(out)
            log_p = log_p + log_p_
            logdet = logdet + det
            z_outs.append(z_new)

        return log_p, logdet, z_outs

    # Ters geçiş işlemi: Ters sırayla katmanlardan geçirerek girdi görüntüsünü yeniden oluşturur
    def reverse(self, z_list, reconstruct=False):
        input = z_list[-1]
        for i, block in enumerate(reversed(self.blocks)):
            input = block.reverse(input, z_list[-(i + 2)], reconstruct=reconstruct)

        return input
