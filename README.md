# Fashion Image Augmentation Pipeline 👗📸

This project provides an **automated image augmentation pipeline** designed for **fashion image datasets**.  
It leverages **OpenCV** and **ImgAug** to perform transformations like flips, rotations, scaling, brightness adjustments, and translations to **create a larger, more diverse dataset** for model training.

---

## 🚀 Features
- Automatically **extracts images** from a zipped dataset
- Performs **image augmentation**:
  - Horizontal & vertical flips
  - Random rotations and scaling
  - Brightness & contrast adjustments
  - Random translations
- Saves the **augmented dataset** to a new folder for easy access

---

## 📂 Project Structure
fashion-image-augmentation-pipeline/
│
├── app.py # Main script for augmentation
├── women-fashion.zip # Example dataset (not included in repo)
├── /output_dataset # Folder where augmented images will be saved
└── README.md # Project documentation


---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashion-image-augmentation-pipeline.git
cd fashion-image-augmentation-pipeline
Install the required Python libraries:
pip install opencv-python imgaug numpy
🖥️ Usage

Prepare your dataset
Place your zip file (e.g., women-fashion.zip) in the project folder.
Run the augmentation script
python app.py
Find your augmented images
All augmented images will be saved in the output_dataset folder.
📈 Example Augmentations

Flip (horizontal & vertical)
Random rotation between -30° and 30°
Scaling between 0.8x and 1.2x
Brightness and contrast adjustments
Translation up to 20% of image size
🔧 Requirements

Python 3.8+
OpenCV
ImgAug
NumPy
🎯 Use Cases

Creating larger fashion image datasets for ML/DL training
Improving model robustness by simulating real-world variations
Experimenting with computer vision projects in the fashion domain
📜 License

This project is released under the MIT License.
You are free to use, modify, and distribute it for educational and commercial purposes.

