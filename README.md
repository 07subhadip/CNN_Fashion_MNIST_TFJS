# Fashion MNIST Magic Blackboard 🎨👕

[![Powered by TensorFlow.js](https://img.shields.io/badge/Powered%20by-TensorFlow.js-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/js)
![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=07subhadip.CNN_Fashion_MNIST_TFJS)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time, interactive web application that recognizes hand-drawn clothing items using Deep Learning models (CNN & FNN) running entirely in the browser via TensorFlow.js.

[🚀 **Try the Live App**](https://07subhadip.github.io/CNN_Fashion_MNIST_TFJS/)

---

## 🔮 Project Overview

The **Magic Blackboard** allows users to draw any of the 10 Fashion MNIST classes (like T-shirts, sneakers, or bags) directly on a digital canvas. The application processes the drawing in real-time, handling resizing, grayscale conversion, and normalization before feeding it into a pre-trained Keras model (converted to TF.js).

Everything runs on the **client-side**. No data is ever sent to a server.

## 📊 Model Performance

This project implements two distinct architectures for comparison.

| Metric                  | CNN (Convolutional) | FNN (Feed-Forward) |
| :---------------------- | :------------------ | :----------------- |
| **Training Accuracy**   | ~91.33%             | ~91.46%            |
| **Training Loss**       | 0.2315              | 0.2359             |
| **Validation Accuracy** | ~89.66%             | ~88.27%            |
| **Validation Loss**     | 0.2886              | 0.3229             |

> **Note:** The CNN model generally outperforms the FNN on complex shapes due to its ability to capture spatial hierarchies.

## 🛠️ Tech Stack

-   **Frontend:** HTML5, CSS3 (Cyberpunk/Dark Theme), JavaScript (ES6+).
-   **Machine Learning:** TensorFlow.js (Inference), Keras (Training).
-   **Tools:** VS Code, Git, GitHub Actions.

## 🏃 How to Run Locally

Since this project loads external model files (`.json` and `.bin`), strict browser CORS policies prevent it from working if you simply open `index.html` as a file. You must use a local server.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/07subhadip/CNN_Fashion_MNIST_TFJS.git
    cd CNN_Fashion_MNIST_TFJS
    ```

2.  **Start a Local Server:**

    -   **Using Python:**
        ```bash
        python -m http.server 8000
        ```
    -   **Using VS Code:** Right-click `index.html` and select **"Open with Live Server"**.

3.  **Open in Browser:**
    Navigate to `http://localhost:8000`.

## 🎨 Supported Classes

The model is trained on the standard Fashion MNIST dataset and recognizes:

1.  T-shirt/top 👕
2.  Trouser 👖
3.  Pullover 🧥
4.  Dress 👗
5.  Coat 🧥
6.  Sandal 👡
7.  Shirt 👔
8.  Sneaker 👟
9.  Bag 👜
10. Ankle boot 🥾

---

<div align="center">
  &copy; 2026 <strong>Subhadip Hensh</strong>. All Rights Reserved.
</div>
