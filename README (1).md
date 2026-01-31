# UrbanSceneClassification
This project implements a CNN in PyTorch for urban scene classification using the MIT Places dataset. It covers preprocessing, model design with Batch Normalization and Dropout, training, validation, and evaluation. Results are visualized, with full GitHub integration for version control and reproducibility.
# Urban Scene Classification with CNN

This project implements a simple Convolutional Neural Network (CNN) for **image classification** using the CIFAR-10 dataset.  
The dataset is automatically downloaded via `torchvision`, so no manual setup is required.

---

## 📂 Project Structure
- `urban_scene_cnn.py` → Main script containing dataset preparation, CNN model, training, and evaluation.
- `data/` → Folder where CIFAR-10 dataset will be downloaded automatically.

---

## ⚙️ Requirements
Make sure you have Python 3.10 (or compatible) installed.  
Install dependencies with:

```bash
pip install torch torchvision matplotlib
How to Run
- Clone or download this project.
- Navigate to the project folder:
cd UrbanSceneClassification
- Activate your virtual environment (if using one):
venv\Scripts\activate
- Run the script:
python urban_scene_cnn.py

Model Architecture
The CNN is lightweight for quick testing:
- Conv2d → 32 filters, kernel size 3, padding 1
- BatchNorm2d → Normalization layer
- ReLU → Activation function
- MaxPool2d → Downsampling
- Dropout (p=0.5) → Regularization
- Fully Connected Layer → Maps features to class outputs

Training & Evaluation
- Dataset: CIFAR-10 (10 classes, 60k images)
- Image size reduced to 64×64 for faster training
- Training uses only 1000 samples for speed
- Validation split: 20% of training set
- Test set: 200 samples
During training, you’ll see validation accuracy per epoch.
After training, the script evaluates on the test set and shows a bar chart of accuracy.

Example Output
Epoch 1, Validation Accuracy: 0.65
Epoch 2, Validation Accuracy: 0.72
Epoch 3, Validation Accuracy: 0.75
Test Accuracy: 0.70

Next Steps
- Increase dataset size for better accuracy (remove Subset restriction).
- Add more convolutional layers for deeper feature extraction.
- Experiment with different optimizers and learning rates.

Notes
- This project is designed for quick CPU testing.
- For real training, use the full CIFAR-10 dataset and larger models.
- GPU acceleration (CUDA) is recommended for faster training.