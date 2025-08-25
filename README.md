# HardHatVision

HardHatVision is a **Computer Vision project** powered by **YOLOv8** for detecting **safety helmets (hard hats)** and **reflective vests** to support workplace safety and compliance monitoring.

---

## 🚀 Features
- Detects **hard hats** in images and videos.
- Detects **reflective safety vests**.
- Real-time inference with YOLOv11.
- Easy retraining on custom datasets.

## 📊 Dataset
This project uses a **custom dataset** in YOLO format (images + labels).  

👉 [Download Dataset](https://drive.google.com/drive/folders/1SrDDSQOcqMCY8e5HnrbaU-jI_Z2HMvls?usp=drive_link)  

---

## 🤖 Model
We fine-tuned **YOLOv11** on the PPE dataset.  

👉 [Download Model](https://drive.google.com/drive/folders/1JMpMdypmEFdpfm66wOnusO3VKSc93HKb?usp=drive_link)  

- Base model: YOLO11s  
- Trained model: Dataset900 base on HardHatVision PPE dataset  

---

## ⚙️ Installation
Clone the repository:
   ```bash
   git clone https://github.com/Yor4-173/HardHatVison.git
   cd HardHatVision
   pip install -r requirement.txt
