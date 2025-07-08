# Human Behavior Recognition Based on Multiscale Convolutional Neural Network (MCNN)

![Taylor & Francis Book Cover](https://github.com/PaladuguSriTeja574/Human_Activity_Recognisation/blob/2a34fa38c48637847656634796d691c71c7e1c97/static/images/Book%20Cover.png)

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
<pre lang="markdown"> 
  ├── model/
    │ ├── mcnn.py # Multiscale CNN architecture
    │ ├── attention_modules.py # ST and DS attention modules
  ├── data/ 
    │ ├── uci_har_dataset/ # UCI Human Activity Recognition dataset 
    │ └── custom_dataset/ # iPhone XR collected custom dataset 
  ├── experiments/ 
    │ ├── train.py # Model training script 
    │ ├── evaluate.py # Model evaluation script
  ├── utils/
    │ ├── preprocessing.py # Data preprocessing and QR decomposition 
  logic ├── docs/
    │ ├── mcnn_diagram.png # MCNN architecture diagram
    │ ├── st_module.png # ST attention module diagram 
  ├── report/ 
    │ ├── documentation.pdf # Detailed technical report of the project 
    │ └── presentation.pptx # Project presentation slides
  ├── requirements.txt # Python package dependencies
  ├── README.md # Project overview and documentation  
</pre>


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
![MCNN Architecture](https://github.com/PaladuguSriTeja574/Human_Activity_Recognisation/blob/33140337e25e0c1a9137fe8114cfaa109f1a6627/static/images/MCNN%20Structure.png) 
![Improved ST Attention](https://github.com/PaladuguSriTeja574/Human_Activity_Recognisation/blob/f2a269ed15389b18c09989b79bfb942e2cce15e3/static/images/ST%20Module.png)

## 📚 References
This work is referenced in the IEEE publication and featured in the Taylor & Francis book:
- 📄 IEEE Article: **Human Behavior Recognition Based on Multiscale CNN**  
      DOI: [10.1109/ACCESS.2022.3209816](https://doi.org/10.1109/ACCESS.2022.3209816)
- 📘 Book Chapter: *Advancing Innovation through AI and Machine Learning Algorithms*  
      Editors: Udara Yedukondalu, V Vijayasri Bolisetty

## 🤝 Authors & Contributors
- 📌 Paladugu Sri Teja Chowdary (corresponding author)
- 🧑‍💻 Collaborators: Muntha Satya Venkata Madav, Uppe Datta Harshitha, Vemagiri Praveen, Sidgam Surya Deepak 
- 🏢 Institution: Vishnu Institute of Technology

## 📫 Contact
For questions or collaborations, feel free to reach out:
- 📧 Email: amithapaladugu6@gmail.com
- 📧 Email: 21pa1a5473@vishnu.edu.in
- 🌐 [LinkedIn](https://www.linkedin.com/in/paladugu-sri-teja-chowdary/)
- 🌍 [Personal Portfolio](https://teja-chowdary-1510.netlify.app/)

## 📝 License
This project is available under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).

---

> 🧠 “Understanding human behavior is not just about data—it’s about context, learning, and intelligent representation.”
