import os
import shutil

# Kaynak klasörlerinizi belirleyin
src_dirs = ['ham10000/HAM10000_images_part_1', 'ham10000/HAM10000_images_part_2']
dest_dir = 'HAM10000_organized'  # Sınıf bazlı düzenlenmiş veriler için hedef klasör

# Etiket dosyasını kontrol edin (HAM10000_metadata.csv)
metadata_file = 'ham10000/HAM10000_metadata.csv'

# Etiketleri oku ve bir mapping oluştur
import pandas as pd
metadata = pd.read_csv(metadata_file)
image_to_class = dict(zip(metadata['image_id'], metadata['dx']))  # Görüntü ID -> Sınıf adı eşleştirme

# Hedef klasörleri oluştur ve görüntüleri taşı
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for src_dir in src_dirs:
    for filename in os.listdir(src_dir):
        image_id, ext = os.path.splitext(filename)
        if image_id in image_to_class:
            class_name = image_to_class[image_id]
            class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            shutil.move(os.path.join(src_dir, filename), os.path.join(class_dir, filename))
