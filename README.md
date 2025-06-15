
# 🌍 Multilingual Language Detection using Machine Learning

This project is a **Multilingual Language Detector** that can identify text written in any of **17 different languages** using Natural Language Processing (NLP) techniques and machine learning model **SVM**


🚀 Try it live here: [**Deployed App Link**](https://multilingual-language-detector.streamlit.app/) 
---


## 📘 About the Project

This application allows users to input a sentence or paragraph in any of the supported languages, and the model will predict which language it is written in.  
It is useful for **preprocessing multilingual data**, **auto language tagging**, and **content categorization** in global platforms.

---

## 🌐 Languages Supported

1. English  
2. Malayalam  
3. Hindi  
4. Tamil  
5. Kannada  
6. French  
7. Spanish  
8. Portuguese  
9. Italian  
10. Russian  
11. Swedish  
12. Dutch  
13. Arabic  
14. Turkish  
15. German  
16. Danish  
17. Greek  

---

## 🛠️ Tech Stack

| Component      | Tools Used                                                              |
| -------------- | ----------------------------------------------------------------------- |
| Data Handling  | `Pandas`, `NumPy`                                                       |
| NLP Processing | `scikit-learn`, `TfidfVectorizer`                                       |
| Models         | `SVM`, `Naive Bayes`, `Logistic Regression`, `Random Forest`, `XGBoost` |
| Web App        | `Streamlit`                                                             |
| Deployment     | `Streamlit Cloud`                                |
| Serialization  | `joblib`                                                                |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/language-detector-app.git
cd language-detector-app
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary><strong>requirements.txt</strong></summary>

```txt
streamlit
scikit-learn
joblib
xgboost
pandas
numpy
```

</details>

### 3. Run the Streamlit app locally

```bash
streamlit run app.py
```

---

## 🧠 Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| SVM (Linear)        | 98.69%   |
| Logistic Regression | 98.3%    |
| Naive Bayes         | 84.4%    |
| Random Forest       | 97.05%   |
| XGBoost             | 96.27%   |

TF-IDF features extracted: **46,987**
Training samples: **10,337**

---

## 🖼️ Screenshots

![image](https://github.com/user-attachments/assets/f9e3e831-500c-42a3-b427-f5e21e180c5b)
![image](https://github.com/user-attachments/assets/5bf459f8-a82d-40ee-a702-979b2c88c969)



---

## 💡 Future Improvements

* Add **Speech-to-Text** support using Whisper or Google Speech API
* Support more languages dynamically via API
* Show **prediction confidence** with a bar chart
* Add **history of predictions**
* Enable **file uploads** (e.g., PDF, TXT) for batch detection

---

## 🧾 License

This project is licensed under the MIT License.

```

