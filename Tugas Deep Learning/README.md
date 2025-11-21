
# README — CIFAR-10 Classification using ViT & Swin Transformer (PyTorch + timm)

Repository ini berisi implementasi lengkap untuk membandingkan performa Vision Transformer (ViT-Base) dan win Transformer (Swin-Tiny)** dalam tugas klasifikasi gambar menggunakan dataset **CIFAR-10**.
Eksperimen dilakukan menggunakan PyTorch, timm, dan torchvision.

---

## 1. Setup Lingkungan dan Import Library

Kode dimulai dengan mengimpor seluruh library yang dibutuhkan:

* `torch`, `torchvision`, `timm` → pemanggilan model & training.
* `sklearn.metrics` → classification report & confusion matrix.
* `matplotlib` dan `seaborn` → visualisasi kurva & matriks kebingungan.
* `tqdm` → progress bar selama training.
* Manajemen device CUDA dan seed.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)
```

Tujuannya agar hasil reproducible dan proses berjalan optimal jika GPU tersedia.

---

## 2. Utility Functions

Beberapa fungsi pendukung disiapkan untuk mempermudah proses:

### **✔ Count Parameters**

Menghitung jumlah parameter total, trainable, dan non-trainable:

```python
def count_params(model):
```

### **✔ Model Size**

Mengukur ukuran model dalam satuan MB dengan menyimpan sementara state-dict:

```python
def model_size_mb(model, path="/tmp/tmp_model.pth"):
```

### **✔ Training History Plot**

Menampilkan kurva *training loss* dan *test accuracy*:

```python
def plot_history(history, title=""):
```

---

## 3. Preprocessing & Augmentasi CIFAR-10

Model ResNet/ViT/Swin pretrained ImageNet membutuhkan input **224×224**, sehingga dilakukan:

* Resize → 256px
* RandomResizedCrop (train)
* CenterCrop (test)
* Normalisasi ImageNet mean/std
* RandomHorizontalFlip untuk augmentasi

```python
train_tf_vit = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
```

Train/test loader dibuat untuk ViT dan Swin secara terpisah.

---

## 4. Training Loop

### **Fungsi Latihan Satu Epoch**

Menggunakan mixed precision (torch.cuda.amp) untuk mempercepat training jika GPU tersedia.

```python
def train_one_epoch(model, loader, optimizer, criterion):
```

Isi proses:

* Set model ke mode `train()`
* Loop batch
* Autocast + scaler untuk AMP
* Compute loss + backward + optimizer step
* Hitung total loss

---

## 5. Evaluasi Model

Tersedia fungsi evaluasi dua versi:

### ** Hitung akurasi**

```python
evaluate_full(model, loader)
```

### **Kembalikan nilai prediksi & label**

Untuk classification report, confusion matrix:

```python
evaluate_full(model, loader, return_preds=True)
```

---

## 6. Pengukuran Kecepatan Inferensi

Inferensi diukur dengan rata-rata waktu per batch:

```python
measure_inference_time(model, loader)
```

Menghasilkan:

* avg ms per gambar
* standar deviasi
* throughput (img/s)

---

## 7. Training ViT-Base (Patch16-224)

Bagian ini memuat training penuh untuk model:

```python
model_vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

Langkahnya:

1. Load model + pindah ke device
2. Hitung parameter & size
3. Training selama 5 epoch
4. Simpan model `.pth`
5. Uji model dan tampilkan:

   * Classification report
   * Confusion matrix
   * Kurva loss & accuracy
   * Inference time

---

## 8. Training Swin-Tiny (Patch4-Window7-224)

Sama seperti ViT, tetapi menggunakan:

```python
model_swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=10)
```

Langkah Tahapan:

* Training 5 epoch
* Evaluasi
* Report
* Confusion matrix
* Kurva performa
* Kecepatan inferensi

---

## 9. Output Eksperimen

Setiap model menghasilkan:

### **1. Jumlah Parameter & Ukuran Model**

* ViT-Base: 85.8M parameter
* Swin-Tiny: 27.5M parameter

### **2. Akurasi**

* ViT-Base ≈ 87.89%
* Swin-Tiny ≈ 93.83%

### **3. Confusion Matrix**

Heatmap dari prediksi model pada 10 kelas CIFAR-10.

### **4. Plot Kurva**

* Training Loss
* Test Accuracy

### **5. Inference Speed**

Contoh output:

```
ViT: 0.86 ms/img
Swin: 2.04 ms/img
```

---

## 10. Kesimpulan Eksperimen

* **Swin-Tiny lebih akurat & lebih efisien parameter**
* **ViT lebih cepat dalam inferensi**
* Perbedaan arsitektur window attention pada Swin membuat generalisasi dataset kecil seperti CIFAR-10 lebih baik.

---



