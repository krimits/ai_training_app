# 📓 Google Colab Notebooks για AI Training

Αυτό το φάκελο περιέχει οδηγίες και links για hands-on εκπαίδευση με Google Colab.

## 🚀 Γρήγορη Έναρξη

### Τι είναι το Google Colab;
Το Google Colaboratory (Colab) είναι ένα δωρεάν cloud-based Jupyter notebook περιβάλλον που σας επιτρέπει να:
- Γράφετε και εκτελείτε Python κώδικα στο browser
- Χρησιμοποιείτε **δωρεάν GPU/TPU** για training
- Συνεργάζεστε σε πραγματικό χρόνο (όπως Google Docs)
- Αποθηκεύετε τα notebooks στο Google Drive σας

### 🎯 Recommended Colab Notebooks

#### 1️⃣ **Beginner Level**

**Linear Regression - Πρόβλεψη Τιμών Σπιτιών**
- 🔗 Link: https://colab.research.google.com/github/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb
- 📚 Τι θα μάθετε: Linear Regression, feature engineering, model evaluation
- ⏱️ Διάρκεια: 30-45 λεπτά
- 🎓 Δυσκολία: Εύκολο

**K-Means Clustering**
- 🔗 Link: https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb
- 📚 Τι θα μάθετε: Unsupervised learning, clustering algorithms
- ⏱️ Διάρκεια: 30 λεπτά
- 🎓 Δυσκολία: Εύκολο

**Decision Trees & Random Forests**
- 🔗 Link: https://colab.research.google.com/github/ageron/handson-ml2/blob/master/06_decision_trees.ipynb
- 📚 Τι θα μάθετε: Tree-based models, ensemble learning
- ⏱️ Διάρκεια: 45 λεπτά
- 🎓 Δυσκολία: Μέτρια

#### 2️⃣ **Intermediate Level**

**Neural Networks με TensorFlow**
- 🔗 Link: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb
- 📚 Τι θα μάθετε: Deep learning basics, TensorFlow/Keras API
- ⏱️ Διάρκεια: 1 ώρα
- 🎓 Δυσκολία: Μέτρια
- 💻 GPU: Συνιστάται

**Convolutional Neural Networks (CNN)**
- 🔗 Link: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb
- 📚 Τι θα μάθετε: Image classification με CNN
- ⏱️ Διάρκεια: 1-1.5 ώρες
- 🎓 Δυσκολία: Μέτρια-Δύσκολη
- 💻 GPU: Απαραίτητη

#### 3️⃣ **Advanced Level**

**NLP με Transformers (BERT)**
- 🔗 Link: https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb
- 📚 Τι θα μάθετε: Transfer learning, fine-tuning BERT
- ⏱️ Διάρκεια: 1.5-2 ώρες
- 🎓 Δυσκολία: Δύσκολη
- 💻 GPU: Απαραίτητη

**Generative Adversarial Networks (GANs)**
- 🔗 Link: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb
- 📚 Τι θα μάθετε: Image generation με GANs
- ⏱️ Διάρκεια: 2+ ώρες
- 🎓 Δυσκολία: Πολύ Δύσκολη
- 💻 GPU: Απαραίτητη

## 🔧 Πώς να Χρησιμοποιήσετε το Colab

### Βήμα 1: Άνοιγμα Notebook
1. Κάντε κλικ σε έναν από τους παραπάνω links
2. Το notebook θα ανοίξει στο Colab
3. Συνδεθείτε με το Google account σας

### Βήμα 2: Ενεργοποίηση GPU (αν χρειάζεται)
1. Πηγαίνετε στο **Runtime** → **Change runtime type**
2. Επιλέξτε **Hardware accelerator**: GPU ή TPU
3. Κάντε κλικ **Save**

### Βήμα 3: Εκτέλεση Κώδικα
- Πατήστε **Shift + Enter** για να τρέξετε ένα cell
- Ή κάντε κλικ στο ▶️ button αριστερά του cell
- Τρέξτε τα cells με τη σειρά!

### Βήμα 4: Αποθήκευση
- **File** → **Save a copy in Drive**
- Το notebook αποθηκεύεται στο "Colab Notebooks" folder στο Drive σας

## 💡 Tips & Best Practices

### 1. GPU Runtime Limits
- Το Colab δίνει δωρεάν GPU για **~12 ώρες** συνεχόμενα
- Μετά από idle time, το session διακόπτεται
- Κάντε **Reconnect** αν χάσετε τη σύνδεση

### 2. Εγκατάσταση Πακέτων
```python
# Τα περισσότερα πακέτα είναι ήδη εγκατεστημένα
# Για νέα πακέτα:
!pip install package_name
```

### 3. Upload/Download Files
```python
# Upload file
from google.colab import files
uploaded = files.upload()

# Download file
files.download('filename.ext')
```

### 4. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 5. Keyboard Shortcuts
- **Ctrl/Cmd + Enter**: Τρέξιμο current cell
- **Shift + Enter**: Τρέξιμο cell + μετακίνηση στο επόμενο
- **Ctrl/Cmd + M + B**: Προσθήκη cell παρακάτω
- **Ctrl/Cmd + M + A**: Προσθήκη cell παραπάνω

## 📚 Επιπλέον Πόροι

### Official Documentation
- [Colab Welcome Notebook](https://colab.research.google.com/notebooks/welcome.ipynb)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/index.html)

### Δημοφιλή Repositories με Colab Notebooks
- [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face NLP Course](https://huggingface.co/course/chapter1)
- [Deep Learning with TensorFlow 2](https://github.com/ageron/handson-ml2)

## 🎓 Learning Path

### Week 1-2: Fundamentals
1. Linear Regression notebook
2. K-Means Clustering notebook
3. Decision Trees notebook

### Week 3-4: Deep Learning Basics
1. Neural Networks με TensorFlow
2. CNN για Image Classification
3. Εφαρμόστε σε δικά σας data

### Week 5-6: Advanced Topics
1. NLP με Transformers
2. GANs για Image Generation
3. Custom project

## ❓ Συχνές Ερωτήσεις

**Q: Είναι το Colab πραγματικά δωρεάν;**
A: Ναι! Η βασική έκδοση είναι 100% δωρεάν. Υπάρχει και Colab Pro ($9.99/μήνα) με περισσότερους πόρους.

**Q: Μπορώ να χρησιμοποιήσω τα δικά μου δεδομένα;**
A: Ναι! Ανεβάστε τα μέσω upload button ή συνδέστε το Google Drive σας.

**Q: Τι γίνεται αν το session timeout;**
A: Τα αποθηκευμένα αρχεία στο Drive παραμένουν, αλλά χάνονται οι μεταβλητές. Τρέξτε ξανά τα cells.

**Q: Μπορώ να χρησιμοποιήσω PyTorch αντί για TensorFlow;**
A: Απολύτως! Και τα δύο frameworks είναι προεγκατεστημένα.

## 🚀 Ready to Start?

Επιλέξτε ένα notebook από την παραπάνω λίστα και ξεκινήστε τη μάθηση!

**Καλή επιτυχία!** 🎉
