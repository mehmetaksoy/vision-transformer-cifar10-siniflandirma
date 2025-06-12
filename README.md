# Vision Transformer ile CIFAR-10 Görüntü Sınıflandırma

Bu proje, **Vision Transformer (ViT)** mimarisini kullanarak CIFAR-10 veri seti üzerinde görüntü sınıflandırma görevini gerçekleştirmektedir. Transformatör mimarilerinin görüntü anlama görevlerindeki etkinliğini göstermek amacıyla geliştirilmiştir.

## 📊 Proje Özeti

- **Model:** google/vit-base-patch16-224 (önceden eğitilmiş)
- **Veri Seti:** CIFAR-10 (50,000 eğitim + 10,000 test görüntüsü)
- **Test Doğruluğu:** %98.83
- **Sınıf Sayısı:** 10 kategori
- **Eğitim Süresi:** ~129 dakika (NVIDIA L4 GPU)

## 🎯 Sınıflandırma Kategorileri

Proje aşağıdaki 10 farklı nesne kategorisini tanıyabilir:

1. **Uçak** (airplane)
2. **Otomobil** (automobile) 
3. **Kuş** (bird)
4. **Kedi** (cat)
5. **Geyik** (deer)
6. **Köpek** (dog)
7. **Kurbağa** (frog)
8. **At** (horse)
9. **Gemi** (ship)
10. **Kamyon** (truck)

## 🚀 Özellikler

- **Vision Transformer (ViT) Mimarisi:** Geleneksel CNN'lere alternatif olarak patch tabanlı görüntü işleme
- **Transfer Öğrenme:** ImageNet üzerinde önceden eğitilmiş model kullanımı
- **Veri Artırma:** RandomResizedCrop ve RandomHorizontalFlip teknikleri
- **Kapsamlı Değerlendirme:** Accuracy, Precision, Recall, F1-Score, AUC metrikleri
- **Görselleştirme:** Karmaşıklık matrisi ve ROC eğrileri
- **Performans Analizi:** Eğitim süreleri ve çıkarım hızı analizi

## 📋 Gereksinimler

```bash
# Temel kütüphaneler
torch>=2.0.0
torchvision>=0.15.0
transformers==4.48.3
datasets==3.6.0

# Veri işleme ve görselleştirme
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
Pillow>=8.0.0

# Notebook ortamı için
tqdm>=4.60.0
```

## 🛠️ Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/kullanici-adi/vision-transformer-cifar10-siniflandirma.git
cd vision-transformer-cifar10-siniflandirma
```

2. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Google Colab için:** Notebook'u doğrudan açıp çalıştırabilirsiniz.

## 📁 Proje Yapısı

```
vision-transformer-cifar10-siniflandirma/
│
├── Vision_Transformer_(ViT)_ile_Görüntü_Sınıflandırma_(CIFAR_10)Kopya.ipynb
├── README.md
├── requirements.txt
├── docs/
│   └── proje_raporu.md
└── results/
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── training_logs.png
```

## 🔄 Kullanım

### Notebook ile Çalıştırma

1. **Google Colab'da açın** veya Jupyter Notebook kullanın
2. **GPU'yu etkinleştirin** (Runtime → Change runtime type → GPU)
3. **Hücreleri sırayla çalıştırın:**
   - Hücre 1: Kütüphane kurulumu
   - Hücre 2: Import ve versiyon kontrolü
   - Hücre 3: CIFAR-10 veri setini yükleme
   - Hücre 4: Veri ön işleme ve augmentasyon
   - Hücre 5: ViT modelini yükleme
   - Hücre 6: Model eğitimi
   - Hücre 7-9: Değerlendirme ve görselleştirme

### Python Scripti ile Çalıştırma

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset

# Veri setini yükle
dataset = load_dataset("cifar10")

# Model ve işlemciyi yükle
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=10)

# Eğitim kodunuz buraya...
```

## 📈 Performans Metrikleri

### Test Seti Sonuçları

| Metrik | Değer |
|--------|--------|
| **Doğruluk (Accuracy)** | %98.83 |
| **Precision (Macro)** | 0.9883 |
| **Recall (Macro)** | 0.9883 |
| **F1-Score (Macro)** | 0.9883 |
| **Specificity (Macro)** | 0.9987 |
| **AUC (Macro)** | 0.9996 |

### Eğitim Detayları

- **Epoch Sayısı:** 10
- **Batch Size:** 32 (eğitim), 64 (değerlendirme)
- **Öğrenme Oranı:** 2e-5
- **Warmup Oranı:** 0.1
- **Weight Decay:** 0.01
- **Toplam Eğitim Süresi:** 129.17 dakika
- **Çıkarım Hızı:** 7,845 görüntü/saniye

## 🧠 Model Mimarisi

**Vision Transformer (ViT)** özellikleri:
- **Patch Boyutu:** 16x16 piksel
- **Girdi Çözünürlüğü:** 224x224 piksel
- **Toplam Parametre:** ~85.8 milyon
- **Önceden Eğitilmiş:** ImageNet dataset
- **Patch Sayısı:** 196 (14x14 grid)

## 📊 Görselleştirmeler

Proje aşağıdaki görselleştirmeleri içerir:

1. **Rastgele Örnek Görüntüler:** Veri setinden örnek görüntüler ve etiketleri
2. **Karmaşıklık Matrisi:** Sınıflar arası tahmin performansı
3. **ROC Eğrileri:** Çok sınıflı AUC analizi
4. **Eğitim Grafikleri:** Epoch bazında kayıp ve metrik değişimleri

## 🔧 Teknik Detaylar

### Veri Ön İşleme
- **Görüntü Boyutlandırma:** 224x224 piksel
- **Normalizasyon:** ImageNet istatistikleri
- **Augmentasyon:** RandomResizedCrop, RandomHorizontalFlip
- **Format Dönüşümü:** PIL → PyTorch tensör

### Model Yapılandırması
- **Transfer Learning:** Önceden eğitilmiş ağırlıklar
- **Fine-tuning:** Tüm katmanlar eğitildi
- **Sınıflandırıcı:** 10 sınıf için yeni başlık
- **Optimizasyon:** AdamW optimizer

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- **Hugging Face:** Transformers ve Datasets kütüphaneleri
- **Google Research:** Vision Transformer modeli
- **CIFAR-10:** Benchmark veri seti
- **PyTorch:** Derin öğrenme framework'ü


## 📞 İletişim

🐛 **Bug Report**: GitHub Issues kullanın  
💡 **Feature Request**: Discussions bölümünden önerinizi paylaşın  
📧 E-posta: [mehmetaksoy49@gmail.com]

- Pull Request ile katkıda bulunun
- Projeyi yıldızlamayı unutmayın! ⭐

---

**Not**: Bu proje eğitim amaçlı geliştirilmiştir ve akademik çalışmalarda referans olarak kullanılabilir.
