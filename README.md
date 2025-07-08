# Human Behavior Recognition Based on Multiscale Convolutional Neural Network (MCNN)

![Taylor & Francis Book Cover](https://user-images.githubusercontent.com/your-link-here/cover.jpg) <!-- Replace with actual image URL if hosting externally -->

> 📘 Published as Chapter 34 in the Taylor & Francis Book:  
> **"Advancing Innovation through AI and Machine Learning Algorithms"**  
> **ISBN:** 978-1-04-116427-2  
> 📌 DOI: [10.1109/ACCESS.2022.3209816](https://doi.org/10.1109/ACCESS.2022.3209816)

## 📌 Abstract
This project addresses the challenge of accurately recognizing human behavior from time-series and video data using a novel **Multiscale Convolutional Neural Network (MCNN)** architecture. It enhances spatiotemporal feature extraction using two innovative attention modules:
- **Space-Time (ST) Interaction Module**
- **Depthwise Separable Convolution Module**

The method performs low-rank learning on segmented video or sensor data and integrates it over time, improving classification performance while maintaining low computational complexity.

## 🚀 Features
- ⚙️ Improved **channel attention** via ST interaction and DS convolution.
- 🔁 Multi-scale CNN that processes input at different receptive fields.
- 📉 Reduced **computational complexity** with **QR decomposition** and **separable convolutions**.
- 🔄 Transferable learning across different network architectures.
- 📈 Validated on **UCI HAR** and **custom real-world datasets**.
- 🧠 Supports deep models like CNN, RNN, LSTM, GRU, BLSTM for comparison.

## 📊 Results
The model outperformed several traditional and deep learning baselines (CNN, LSTM, GRU, SVM) with significant accuracy improvement:
- ✅ Increased accuracy with 2-stage and 3-stage progressive supervision
- ⏱️  Faster convergence and reduced training overhead

## 📂 Project Structure
├── model/
│ ├── mcnn.py # Multiscale CNN architecture
│ ├── attention_modules.py # ST & DS attention modules
├── data/
│ ├── uci_har_dataset/ # UCI Human Activity Recognition Dataset
│ └── custom_dataset/ # Custom iPhone XR collected dataset
├── experiments/
│ ├── train.py # Training pipeline
│ ├── evaluate.py # Evaluation and comparison
├── utils/
│ ├── preprocessing.py # Sliding window, normalization, QR decomposition
├── README.md
├── requirements.txt
└── report/
├── documentation.pdf # Detailed technical report
└── presentation.pptx # Project presentation slides

markdown
Copy
Edit

## 🧪 Datasets
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- Custom dataset using iPhone XR (4200 samples, 102 features, balanced & unbalanced)

## 🛠️ Tech Stack
- **Python 3.7+** (Anaconda / Jupyter / Google Colab)
- **TensorFlow** (Deep learning backend)
- **MATLAB** (for specific preprocessing and simulation)
- **NumPy, Pandas, Scikit-learn** (data processing)
- **OpenCV** (optional for video input)

## 🖥️ System Requirements
- OS: Windows / Linux
- CPU: Intel i3 or higher
- RAM: 4 GB minimum
- Disk: 250 GB minimum

## 📷 Sample Visuals
![MCNN Architecture](docs/mcnn_diagram.png) <!-- Add diagram if available -->
![Improved ST Attention](docs/st_module.png)

## 📚 References
This work is referenced in the IEEE publication and featured in the Taylor & Francis book:
- 📄 IEEE Article: **Human Behavior Recognition Based on Multiscale CNN**  
  DOI: [10.1109/ACCESS.2022.3209816](https://doi.org/10.1109/ACCESS.2022.3209816)
- 📘 Book Chapter: *Advancing Innovation through AI and Machine Learning Algorithms*  
  Editors: Udara Yedukondalu, V Vijayasri Bolisetty

## 🤝 Authors & Contributors
- 📌 Your Name (corresponding author / contributor)
- 🧑‍💻 Collaborators: Include names or GitHub links
- 🏢 Institution: Mention your college/university

## 📫 Contact
For questions or collaborations, feel free to reach out:
- 📧 Email: yourname@example.com
- 🌐 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🌍 [Personal Portfolio](https://yourportfolio.com)

## 📝 License
This project is available under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).

---

> 🧠 “Understanding human behavior is not just about data—it’s about context, learning, and intelligent representation.”
