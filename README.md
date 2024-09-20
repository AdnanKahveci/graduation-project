# Glow Model Training

Bu proje, Glow modelini kullanarak görüntü verileri üzerinde generative model (üretici model) eğitimi yapmayı amaçlar. Glow, tersinir bir generative flow modelidir ve farklı veri setleriyle çalışabilir. Bu rehber, Glow modelinin nasıl eğitileceğini ve nasıl test edileceğini açıklar.

## Gereksinimler

Bu projeyi çalıştırmadan önce aşağıdaki yazılım ve kütüphanelerin sisteminizde kurulu olduğundan emin olun:

- Python 3.6+
- CUDA (isteğe bağlı, GPU kullanıyorsanız)
- PyTorch
- torchvision
- tqdm
- PIL (Pillow)

Aşağıdaki komut ile gerekli Python paketlerini yükleyebilirsiniz:

```bash
pip install torch torchvision tqdm Pillow
```
# Proje Dosyaları
- train.py: Modeli eğitmek için kullanılan ana dosya.
- model.py: Glow modelinin tanımlandığı dosya.
- sample/: Eğitim sırasında üretilen örneklerin kaydedileceği dizin.
- checkpoint/: Model ve optimizasyon parametrelerinin kaydedileceği dizin.

# Kullanım
Glow modelini eğitmek için aşağıdaki adımları izleyin:

## 1. Veriyi Hazırlama
İlk olarak, Glow modelini eğitmek istediğiniz görüntü veri setini indirin ve yerel bir dizine kaydedin. `ImageFolder` formatında bir veri kümesine sahip olmanız gerekiyor.

Aşağıdaki bağlantıdan **Skin Cancer MNIST: HAM10000** veri setini indirebilirsiniz:

- [Skin Cancer MNIST: HAM10000 Veri Seti](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)

Veri setinizin dizin yapısı aşağıdaki gibi olmalıdır:

    PATH/
    |
    class1/
    |   |
    |   - image1.jpg
    |   - image2.jpg
    |   - ...
    |
    class2/
    |   |
    |   - image1.jpg
    |   - image2.jpg
    |   - ...
    |
    ...
Eğer sadece bir sınıf kullanıyorsanız, yine de dizin yapısı yukarıdaki gibi olmalıdır (en az bir alt klasör gereklidir).

## Komut Satırı Argümanları

| Argüman | Açıklama | Varsayılan Değer |
|---------|----------|------------------|
| `PATH` | Görüntülerin bulunduğu dizin | - |
| `--batch` | Batch (küme) boyutu | 16 |
| `--iter` | Eğitim sırasında yapılacak maksimum iterasyon sayısı | 200000 |
| `--n_flow` | Her bloktaki flow (akış) sayısı | 32 |
| `--n_block` | Modeldeki blok sayısı | 4 |
| `--no_lu` | LU ayrışımı kullanmamak için (varsayılan olarak LU ayrışımı kullanılır) | - |
| `--affine` | Additive coupling yerine affine coupling kullanır | - |
| `--n_bits` | Görüntülerin bit sayısı | 5 |
| `--lr` | Öğrenme oranı | 1e-4 |
| `--img_size` | Görüntü boyutu (genişlik ve yükseklik) | 64 |
| `--temp` | Örnekleme sırasında kullanılan sıcaklık | 0.7 |
| `--n_sample` | Örnekleme sırasında üretilecek görüntü sayısı | 20 |

## 2. Eğitim Sırasında Çıktılar
Eğitim sırasında, modelin ürettiği örnek görüntüler her 100 iterasyonda bir `sample/` dizinine kaydedilecektir. Ayrıca, model ve optimizasyon parametreleri her 10.000 iterasyonda bir `checkpoint/` dizinine kaydedilecektir.
## 3. Modeli Kaydetme
Model ve optimizasyon parametreleri belirli aralıklarda (10.000 iterasyon) `checkpoint/` dizinine .pt formatında kaydedilecektir. Bu dosyalar gelecekteki eğitim aşamalarında ya da test aşamalarında kullanılabilir.
# Örnek Eğitim Komutu
Örneğin, HAM10000 veri setini kullanarak Glow modelini eğitmek için şu komutu kullanabilirsiniz:
```bash
python train.py HAM10000
```
# Modeli Kaydedilen Kontrol Noktalarından Yüklemek
Eğitimi belirli bir noktada durdurduysanız ve devam etmek istiyorsanız, kaydedilen model ve optimizasyon parametrelerini şu şekilde yükleyebilirsiniz:
```bash
model.load_state_dict(torch.load("checkpoint/model_100000.pt"))
optimizer.load_state_dict(torch.load("checkpoint/optim_100000.pt"))
```
