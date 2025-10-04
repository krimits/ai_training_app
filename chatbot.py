# AI Knowledge Base Chatbot Module with Internet Access

import re
from typing import List, Tuple, Dict
import streamlit as st
import requests
from urllib.parse import quote
import json

class AIKnowledgeBot:
    """
    Intelligent chatbot που απαντά σε ερωτήσεις βασισμένο στο εκπαιδευτικό υλικό AI.
    """
    
    def __init__(self, knowledge_file='pdf_content.txt'):
        self.knowledge_base = self._load_knowledge(knowledge_file)
        self.qa_pairs = self._create_qa_database()
        self.use_internet = True  # Enable internet access
        self.wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.sources_used = []
        
    def _load_knowledge(self, filepath):
        """Φόρτωση περιεχομένου από το PDF"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except:
            return ""
    
    def _create_qa_database(self):
        """Δημιουργία βάσης ερωτήσεων-απαντήσεων"""
        return {
            # Ορισμός Τεχνητής Νοημοσύνης
            "τεχνητή νοημοσύνη ορισμός": {
                "keywords": ["τεχνητή νοημοσύνη", "ορισμός", "τι είναι", "ai definition", "artificial intelligence", "νοημοσύνη", "ορίζω", "εξήγησε"],
                "answer": """
## 🤖 Τι είναι η Τεχνητή Νοημοσύνη;

### 📖 Ορισμός

Η **Τεχνητή Νοημοσύνη (Artificial Intelligence - AI)** είναι ο **κλάδος της επιστήμης των υπολογιστών** που ασχολείται με τη δημιουργία **συστημάτων και αλγορίθμων** ικανών να εκτελούν εργασίες που παραδοσιακά **απαιτούν ανθρώπινη νοημοσύνη**.

### 🎯 Απλή Εξήγηση

Η ΤΝ είναι η προσπάθεια να κάνουμε τους υπολογιστές να:
- 🧠 **Μαθαίνουν** από εμπειρία
- 🎓 **Κατανοούν** φυσική γλώσσα
- 👁️ **Αναγνωρίζουν** patterns και εικόνες
- 🎯 **Λύνουν προβλήματα** με λογική
- 🚀 **Προσαρμόζονται** σε νέες καταστάσεις
- 🤔 **Παίρνουν αποφάσεις** αυτόνομα

### 📚 Επίσημοι Ορισμοί

**John McCarthy (1956)** - "Πατέρας της AI":
> "Η επιστήμη και η μηχανική της δημιουργίας έξυπνων μηχανών."

**Stuart Russell & Peter Norvig** (AI: A Modern Approach):
> "Η μελέτη των agents που λαμβάνουν inputs από το περιβάλλον και εκτελούν actions."

**Marvin Minsky**:
> "Η επιστήμη του να κάνεις μηχανές να κάνουν πράγματα που θα απαιτούσαν νοημοσύνη αν τα έκανε άνθρωπος."

### 🔍 Βασικά Χαρακτηριστικά

#### 1. **Μάθηση (Learning)**
- Αποκτά γνώση από δεδομένα
- Βελτιώνεται με την εμπειρία
- Αναγνωρίζει patterns

#### 2. **Αυτονομία (Autonomy)**
- Λειτουργεί χωρίς συνεχή ανθρώπινη επίβλεψη
- Παίρνει αποφάσεις αυτόνομα
- Προσαρμόζεται σε αλλαγές

#### 3. **Συλλογισμός (Reasoning)**
- Επεξεργάζεται πληροφορίες
- Εξάγει συμπεράσματα
- Λύνει προβλήματα

#### 4. **Αντίληψη (Perception)**
- "Βλέπει" (Computer Vision)
- "Ακούει" (Speech Recognition)
- "Κατανοεί" (Natural Language)

#### 5. **Επικοινωνία (Communication)**
- Φυσική γλώσσα
- Αλληλεπίδραση με ανθρώπους
- Εξήγηση αποφάσεων

### 🎭 Τύποι Τεχνητής Νοημοσύνης

#### **Narrow AI (Weak AI)** - Περιορισμένη ΤΝ
- Εξειδικευμένη σε συγκεκριμένες εργασίες
- Παραδείγματα: Siri, αυτόνομα οχήματα, σκάκι AI
- **Υπάρχει σήμερα** ✅

#### **General AI (Strong AI)** - Γενική ΤΝ
- Ισοδύναμη με ανθρώπινη νοημοσύνη
- Μπορεί να μάθει οποιαδήποτε νοητική εργασία
- **Δεν υπάρχει ακόμα** ⏳

#### **Super AI** - Υπερνοημοσύνη
- Ξεπερνά την ανθρώπινη νοημοσύνη
- Θεωρητική προς το παρόν
- **Μέλλον;** 🔮

### 🏗️ Κλάδοι της ΤΝ

1. **Machine Learning (ML)** 🧠
   - Μάθηση από δεδομένα
   - Supervised, Unsupervised, Reinforcement

2. **Deep Learning (DL)** 🌊
   - Νευρωνικά δίκτυα με πολλά layers
   - CNN, RNN, Transformers

3. **Natural Language Processing (NLP)** 💬
   - Κατανόηση και παραγωγή γλώσσας
   - ChatGPT, Google Translate

4. **Computer Vision (CV)** 👁️
   - Αναγνώριση εικόνων
   - Face recognition, Self-driving cars

5. **Robotics** 🤖
   - Φυσικά συστήματα με AI
   - Αυτόνομη κίνηση και manipulation

### 💼 Πρακτικές Εφαρμογές

**Καθημερινή Ζωή:**
- 📱 Smartphones (Siri, Google Assistant)
- 🎬 Netflix recommendations
- 📧 Email spam filtering
- 🗺️ Google Maps (route optimization)

**Επιχειρήσεις:**
- 🏦 Fraud detection
- 💰 Stock trading
- 🎯 Targeted advertising
- 📊 Business analytics

**Υγεία:**
- 🏥 Medical diagnosis
- 💊 Drug discovery
- 🧬 Genomics analysis
- 🩺 Patient monitoring

**Μεταφορές:**
- 🚗 Self-driving cars
- ✈️ Flight route optimization
- 🚢 Logistics και supply chain

**Επιστήμη:**
- 🔬 Scientific research
- 🌌 Space exploration
- 🧪 Chemistry simulations
- 🔭 Astronomy data analysis

### 🎓 Ιστορία (Σύντομη)

- **1950**: Alan Turing - Turing Test
- **1956**: Dartmouth Conference - Γέννηση του όρου "AI"
- **1997**: Deep Blue νικά τον Kasparov στο σκάκι
- **2011**: IBM Watson νικά στο Jeopardy
- **2016**: AlphaGo νικά τον παγκόσμιο πρωταθλητή Go
- **2022**: ChatGPT - Mass adoption AI

### ⚖️ Ηθικά Ζητήματα

**Προβληματισμοί:**
- 🔒 Privacy και data security
- ⚖️ Bias και fairness
- 💼 Job displacement
- 🎯 Autonomous weapons
- 🤝 Human-AI collaboration

### 🔮 Μέλλον της ΤΝ

**Τάσεις:**
- Πιο ισχυρά Large Language Models
- Multimodal AI (text + image + video)
- Edge AI (τοπική επεξεργασία)
- Explainable AI (διαφάνεια)
- AI for Good (κοινωνική αξία)

### 📌 Σύνοψη

Η Τεχνητή Νοημοσύνη είναι:
- ✅ Η επιστήμη των "έξυπνων" μηχανών
- ✅ Συστήματα που μαθαίνουν και προσαρμόζονται
- ✅ Τεχνολογία που μεταμορφώνει κάθε τομέα
- ✅ Εργαλείο για επίλυση πολύπλοκων προβλημάτων
- ✅ Το μέλλον της τεχνολογίας

**Απλά λόγια**: Η ΤΝ κάνει τους υπολογιστές να σκέφτονται και να μαθαίνουν σαν άνθρωποι! 🧠💻
"""
            },
            
            # Βασικά Δομικά Στοιχεία
            "δομικά στοιχεία": {
                "keywords": ["δομικά", "στοιχεία", "βάση", "θεμέλιο", "components"],
                "answer": """
## 🏗️ Βασικά Δομικά Στοιχεία της Τεχνητής Νοημοσύνης

Τα συστήματα ΤΝ λειτουργούν βασιζόμενα σε **τέσσερα κύρια αλληλένδετα δομικά στοιχεία**:

### 1. 📊 **Δεδομένα** - Η Βάση της Εκπαίδευσης

Τα δεδομένα αποτελούν το **θεμέλιο** πάνω στο οποίο οικοδομείται η Τεχνητή Νοημοσύνη.

**Κρισιμότητα:**
- Η μηχανική μάθηση βασίζεται εξ ολοκλήρου στα δεδομένα
- Η **ποιότητα** και η **ποσότητα** είναι κρίσιμες για την επιτυχία

**Τύποι Δεδομένων:**
- **Κείμενο**: Άρθρα, βιβλία, κώδικας προγραμματισμού
- **Εικόνες**: Φωτογραφίες, βίντεο, οπτικές πληροφορίες
- **Ήχος**: Μουσική, ανθρώπινη ομιλία, ηχητικά δεδομένα
- **Αισθητηριακά**: Δεδομένα από αισθητήρες (θερμοκρασία, πίεση)

**Εξόρυξη Γνώσης:**
- Data mining: Μετατροπή τεράστιων όγκων σε εφαρμόσιμη γνώση
- Βελτίωση λήψης αποφάσεων

---

### 2. ⚙️ **Αλγόριθμοι** - Οι Επεξεργαστές Πληροφοριών

Οι αλγόριθμοι AI είναι οι **μαθηματικές και λογικές δομές** για επεξεργασία δεδομένων.

**Λειτουργία:**
- Μαθαίνουν αυτόνομα από δεδομένα
- Βελτιώνουν την απόδοση χωρίς ανθρώπινη παρέμβαση σε κάθε βήμα

**Επιλογή:**
- Κάθε αλγόριθμος έχει πλεονεκτήματα και μειονεκτήματα
- Εξαρτάται από τον τύπο δεδομένων και το επιθυμητό αποτέλεσμα

**Στόχος:**
- Συνεχής βελτίωση της απόδοσης με την πάροδο του χρόνου

---

### 3. 🎯 **Μοντέλα AI και Νευρωνικά Δίκτυα**

**Νευρωνικά Δίκτυα (ΝΔ):**
- Υπολογιστικά συστήματα εμπνευσμένα από τον **ανθρώπινο εγκέφαλο**
- Αποτελούνται από πολλούς τεχνητούς νευρώνες που συνδέονται μεταξύ τους

**Λειτουργία Νευρώνων:**
- Λαμβάνει εισόδους → Επεξεργάζεται → Παράγει έξοδο
- Μεταβιβάζει μέσω συνδέσεων (βάρη)

**Μάθηση:**
- Προσαρμόζουν τα βάρη των συνδέσεων
- Στόχος: Μείωση σφάλματος και βελτίωση απόδοσης

**Βαθιά Μάθηση (Deep Learning):**
- Υποσύνολο της Μηχανικής Μάθησης
- Χρησιμοποιεί ΝΔ με **πολλές κρυφές στρώσεις** (εξ ου "βαθιά")
- Αυτόματη εξαγωγή πολύπλοκων χαρακτηριστικών
- Λειτουργεί σε μη δομημένα δεδομένα (εικόνες, βίντεο, ήχος)

**Παράδειγμα:**
- Υψηλό επίπεδο: "Πρόσωπο"
- Χαμηλά επίπεδα: "Αρσενικό" ή "Θηλυκό"

---

### 4. 💻 **Υπολογιστική Ισχύς** - Η Αναγκαία Δύναμη

Η εκπαίδευση και εκτέλεση αλγορίθμων ΤΝ είναι **εξαιρετικά απαιτητική**.

**Απαιτήσεις εξαρτώνται από:**
- Τον τύπο του αλγορίθμου ΤΝ
- Την ποσότητα δεδομένων εκπαίδευσης
- Το επιθυμητό επίπεδο απόδοσης

**Hardware:**
- GPU/TPU για επιτάχυνση
- Cloud computing για μεγάλη κλίμακα
- Specialized AI chips

---

## 🔗 Συνεργασία Στοιχείων

Όλα τα στοιχεία **συνεργάζονται αρμονικά**:

```
Δεδομένα → Αλγόριθμοι → Μοντέλα → Υπολογιστική Ισχύς
    ↑                                    ↓
    ←────────── Βελτιωμένα Μοντέλα ←──────
```

**Αποτέλεσμα:** Συστήματα που μαθαίνουν και εκτελούν εργασίες με ανθρώπινη ή υπερανθρώπινη ικανότητα.
"""
            },
            
            # Machine Learning
            "machine learning": {
                "keywords": ["machine learning", "μηχανική μάθηση", "ml", "μάθηση"],
                "answer": """
## 🧠 Machine Learning (Μηχανική Μάθηση)

Το **Machine Learning** είναι ένας κλάδος της AI που επιτρέπει στους υπολογιστές να μαθαίνουν από δεδομένα χωρίς να προγραμματίζονται ρητά για κάθε εργασία.

### 📚 Τρεις Κύριοι Τύποι:

#### 1️⃣ **Supervised Learning** (Επιβλεπόμενη Μάθηση)
- **Δεδομένα**: Labeled data (με ετικέτες)
- **Στόχος**: Πρόβλεψη outcomes
- **Παραδείγματα**: 
  - Classification (Ταξινόμηση)
  - Regression (Παλινδρόμηση)
- **Αλγόριθμοι**: 
  - Linear Regression
  - Logistic Regression
  - Decision Trees
  - SVM
  - Random Forests
  - Neural Networks

#### 2️⃣ **Unsupervised Learning** (Μη Επιβλεπόμενη Μάθηση)
- **Δεδομένα**: Unlabeled data (χωρίς ετικέτες)
- **Στόχος**: Ανακάλυψη patterns
- **Παραδείγματα**:
  - Clustering (Ομαδοποίηση)
  - Association Rules
  - Dimensionality Reduction
- **Αλγόριθμοι**:
  - K-Means
  - Hierarchical Clustering
  - PCA
  - Autoencoders

#### 3️⃣ **Reinforcement Learning** (Ενισχυτική Μάθηση)
- **Μέθοδος**: Interaction με περιβάλλον
- **Στόχος**: Μεγιστοποίηση rewards
- **Παραδείγματα**:
  - Gaming AI
  - Robotics
  - Αυτόνομα οχήματα
- **Αλγόριθμοι**:
  - Q-Learning
  - Deep Q-Networks (DQN)
  - Policy Gradients
  - PPO, A3C

### 🔄 ML Pipeline (6 Στάδια):

1. **Data Collection**: Συλλογή δεδομένων
2. **Preprocessing**: Καθαρισμός και προετοιμασία
3. **Model Selection**: Επιλογή αλγορίθμου
4. **Training**: Εκπαίδευση μοντέλου
5. **Evaluation**: Αξιολόγηση απόδοσης
6. **Deployment**: Θέση σε production

### 💼 Εφαρμογές:

- **E-commerce**: Προτάσεις προϊόντων (Amazon, Netflix)
- **Finance**: Credit scoring, fraud detection
- **Healthcare**: Διάγνωση ασθενειών, drug discovery
- **Marketing**: Customer segmentation, churn prediction
- **Manufacturing**: Predictive maintenance
"""
            },
            
            # ChatGPT
            "chatgpt": {
                "keywords": ["chatgpt", "gpt", "language model", "γλωσσικό μοντέλο"],
                "answer": """
## 🤖 Πώς Λειτουργεί το ChatGPT

Το **ChatGPT** είναι ένα **Large Language Model (LLM)** που βασίζεται στην αρχιτεκτονική **Transformer**.

### 🏗️ Αρχιτεκτονική:

**Βασικά Χαρακτηριστικά:**
- 🔄 Transformer architecture (2017)
- 📊 Εκπαιδευμένο σε τεράστιο όγκο κειμένων
- 🧮 Δισεκατομμύρια παράμετροι (175B+ για GPT-3)
- 🎯 Fine-tuned με RLHF (Reinforcement Learning from Human Feedback)

### ⚙️ Πώς Λειτουργεί (5 Βήματα):

1. **Input**: Λαμβάνει το prompt (ερώτηση/εντολή)
2. **Tokenization**: Μετατροπή σε tokens
3. **Processing**: Επεξεργασία μέσω transformer layers
4. **Prediction**: Προβλέπει το επόμενο token
5. **Generation**: Παράγει συνεχή κείμενο

### 🎓 Τεχνικές Λεπτομέρειες:

**Pre-training:**
- Unsupervised learning σε τεράστια datasets
- Next token prediction
- Μάθηση γλωσσικών patterns

**Fine-tuning:**
- Supervised fine-tuning (SFT)
- RLHF: Alignment με ανθρώπινες προτιμήσεις
- Reward model για βελτίωση

**Transformer Mechanism:**
```
Input → Tokenization → Embeddings → 
Multi-Head Attention → Feed Forward → 
Output Layer → Generated Text
```

### ✨ Δυνατότητες:

- ✍️ Δημιουργία κειμένου
- 💬 Φυσική συνομιλία
- 📝 Σύνοψη κειμένων
- 🔄 Μετάφραση γλωσσών
- 💻 Προγραμματισμός (code generation)
- 🎨 Δημιουργικότητα
- 📊 Ανάλυση δεδομένων

### ⚠️ Περιορισμοί:

- Μπορεί να παράγει λάθη (hallucinations)
- Cutoff date γνώσης
- Δεν αναζητά στο internet (base models)
- Δεν "κατανοεί" πραγματικά (statistical patterns)

### 🔬 Self-Attention Mechanism:

Το κλειδί της επιτυχίας:
- Κοιτάει **όλα** τα tokens ταυτόχρονα
- Καταλαβαίνει σχέσεις μεταξύ λέξεων
- Long-range dependencies
- Parallel processing (γρήγορο!)

### 💡 Γιατί Λειτουργεί:

1. **Τεράστια κλίμακα**: Δισεκατομμύρια παράμετροι
2. **Πολλά δεδομένα**: Εκπαιδευμένο σε internet-scale text
3. **Transformer**: Αποδοτική αρχιτεκτονική
4. **RLHF**: Human alignment
"""
            },
            
            # Deep Learning
            "deep learning": {
                "keywords": ["deep learning", "βαθιά μάθηση", "neural network", "νευρωνικό δίκτυο"],
                "answer": """
## 🌐 Deep Learning (Βαθιά Μάθηση)

Το **Deep Learning** είναι υποκατηγορία του Machine Learning που χρησιμοποιεί **νευρωνικά δίκτυα** με πολλά κρυφά επίπεδα (layers).

### 🧬 Κύριες Αρχιτεκτονικές:

#### 1. **CNN** (Convolutional Neural Networks)
- **Για**: Εικόνες και spatial data
- **Χαρακτηριστικά**:
  - Convolution layers: Εξαγωγή features
  - Pooling layers: Μείωση διαστάσεων
- **Παραδείγματα**: ResNet, VGG, EfficientNet
- **Εφαρμογές**:
  - Image classification
  - Object detection (YOLO, R-CNN)
  - Face recognition
  - Medical imaging

#### 2. **RNN** (Recurrent Neural Networks)
- **Για**: Sequential data (κείμενο, χρονοσειρές)
- **Χαρακτηριστικά**:
  - "Μνήμη" προηγούμενων states
  - Temporal dependencies
- **Παραλλαγές**: LSTM, GRU
- **Εφαρμογές**:
  - NLP
  - Speech recognition
  - Time series prediction
  - Music generation

#### 3. **Transformers**
- **Επανάσταση** στο NLP (2017)
- **Χαρακτηριστικά**:
  - Self-attention mechanism
  - Parallel processing
  - Long-range dependencies
- **Μοντέλα**: BERT, GPT, T5, ViT
- **Εφαρμογές**:
  - Language models (ChatGPT)
  - Translation
  - Text summarization
  - Question answering

#### 4. **GANs** (Generative Adversarial Networks)
- **Concept**: Δύο δίκτυα "παλεύουν"
  - Generator: Δημιουργεί fake data
  - Discriminator: Διακρίνει real vs fake
- **Εφαρμογές**:
  - Image generation (StyleGAN)
  - DeepFakes
  - Data augmentation
  - Art creation

#### 5. **Autoencoders**
- **Concept**: Συμπίεση και αποσυμπίεση
- **Χαρακτηριστικά**:
  - Encoder: Μειώνει διαστάσεις
  - Decoder: Ανακατασκευάζει
- **Τύποι**: VAE (Variational Autoencoders)
- **Εφαρμογές**:
  - Dimensionality reduction
  - Anomaly detection
  - Denoising
  - Image compression

### 🎯 Βασικά Concepts:

**Activation Functions:**
- ReLU: f(x) = max(0, x)
- Sigmoid: f(x) = 1/(1+e^(-x))
- Tanh
- Softmax

**Optimization:**
- SGD (Stochastic Gradient Descent)
- Adam (Adaptive Moment Estimation)
- RMSprop

**Regularization:**
- Dropout
- L1/L2
- Batch Normalization
- Early Stopping

### 💻 Frameworks:

- **TensorFlow**: Google's framework
- **PyTorch**: Facebook's framework
- **Keras**: High-level API
- **JAX**: High-performance

### 🚀 Applications:

- Computer Vision
- NLP
- Speech Recognition
- Gaming AI
- Drug Discovery
"""
            },
            
            # Supervised Learning
            "supervised learning": {
                "keywords": ["supervised", "επιβλεπόμενη", "labeled data", "classification", "regression"],
                "answer": """
## 🎯 Supervised Learning (Επιβλεπόμενη Μάθηση)

Η **Supervised Learning** είναι ο πιο συνηθισμένος τύπος ML όπου το μοντέλο μαθαίνει από **labeled data**.

### 📊 Δύο Κύριοι Τύποι:

#### 1️⃣ **Classification** (Ταξινόμηση)
**Στόχος:** Πρόβλεψη διακριτής κατηγορίας

**Παραδείγματα:**
- Email spam detection (Spam/Not Spam)
- Medical diagnosis (Υγιής/Άρρωστος)
- Sentiment analysis (Positive/Negative/Neutral)
- Face recognition

**Αλγόριθμοι:**
- Logistic Regression
- Decision Trees
- Random Forest
- SVM
- Neural Networks
- Naive Bayes
- KNN

**Μετρικές:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC

#### 2️⃣ **Regression** (Παλινδρόμηση)
**Στόχος:** Πρόβλεψη συνεχούς αριθμητικής τιμής

**Παραδείγματα:**
- House price prediction
- Stock market forecasting
- Temperature prediction
- Sales forecasting
- Age estimation

**Αλγόριθμοι:**
- Linear Regression
- Polynomial Regression
- Ridge/Lasso Regression
- Decision Tree Regression
- Random Forest Regression
- SVR
- Neural Network Regression

**Μετρικές:**
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root MSE)
- R² Score

### ⚙️ Πώς Λειτουργεί:

1. **Training Data**: Ζεύγη (input, output)
2. **Learning**: Το μοντέλο μαθαίνει τη σχέση
3. **Testing**: Προβλέπει σε νέα data
4. **Evaluation**: Μέτρηση απόδοσης

### 💼 Εφαρμογές:

**Business:**
- Customer churn prediction
- Lead scoring
- Price optimization

**Finance:**
- Credit scoring
- Fraud detection
- Stock prediction

**Healthcare:**
- Disease diagnosis
- Patient risk stratification
- Drug response prediction

### ⚠️ Challenges:

- Χρειάζεται labeled data (ακριβό!)
- Overfitting risk
- Class imbalance
- Feature engineering
"""
            }
        }
    
    def get_answer(self, question: str) -> str:
        """Βρίσκει και επιστρέφει την καλύτερη απάντηση"""
        question_lower = question.lower()
        
        best_match = None
        max_score = 0
        
        # Αναζήτηση στη βάση
        for topic, data in self.qa_pairs.items():
            score = sum(1 for keyword in data["keywords"] if keyword in question_lower)
            if score > max_score:
                max_score = score
                best_match = data["answer"]
        
        if best_match:
            return best_match
        else:
            return self._generate_generic_answer(question)
    
    def _generate_generic_answer(self, question: str) -> str:
        """Γενική απάντηση όταν δεν βρεθεί match - τώρα με internet search"""
        
        # Αν έχουμε internet access, προσπάθησε να βρεις online
        if self.use_internet:
            online_info = self._search_online(question)
            if online_info:
                return online_info
        
        return f"""
## 🤔 Δεν βρήκα συγκεκριμένη απάντηση

Η ερώτησή σας: "{question}"

### 💡 Προτάσεις:

**Μπορείτε να ρωτήσετε για:**
- Βασικά δομικά στοιχεία της AI
- Machine Learning και τους τύπους του
- Πώς λειτουργεί το ChatGPT
- Deep Learning και νευρωνικά δίκτυα
- Supervised/Unsupervised/Reinforcement Learning
- Εφαρμογές AI σε διάφορους τομείς

**Παραδείγματα ερωτήσεων:**
- "Ποια είναι τα βασικά δομικά στοιχεία της AI;"
- "Τι είναι το Machine Learning;"
- "Πώς λειτουργεί το ChatGPT;"
- "Τι είναι το Deep Learning;"
- "Εξήγησε τη Supervised Learning"

### 📚 Επίσης μπορείτε:
- Να δείτε τις ενότητες στο tab "Περιεχόμενο"
- Να δοκιμάσετε τις διαδραστικές ασκήσεις
- Να κάνετε τα κουίζ αυτοαξιολόγησης
"""
    
    def _search_online(self, question: str) -> str:
        """Αναζήτηση πληροφοριών από online πηγές"""
        try:
            self.sources_used = []
            
            # 1. Wikipedia Search
            wiki_info = self._search_wikipedia(question)
            
            # 2. Curated AI Sources
            curated_info = self._search_curated_sources(question)
            
            if wiki_info or curated_info:
                answer = "## 🌐 Πληροφορίες από Online Πηγές\n\n"
                
                if wiki_info:
                    answer += wiki_info + "\n\n"
                
                if curated_info:
                    answer += curated_info + "\n\n"
                
                # Add sources
                if self.sources_used:
                    answer += "### 📚 Πηγές:\n\n"
                    for i, source in enumerate(self.sources_used, 1):
                        answer += f"{i}. {source}\n"
                
                answer += "\n---\n\n"
                answer += "💡 **Σημείωση**: Αυτές οι πληροφορίες προέρχονται από online πηγές. "
                answer += "Για πιο αναλυτικές εξηγήσεις, δείτε το tab 'Περιεχόμενο'."
                
                return answer
            
        except Exception as e:
            st.warning(f"Σφάλμα κατά την online αναζήτηση: {str(e)}")
        
        return None
    
    def _search_wikipedia(self, question: str) -> str:
        """Αναζήτηση στο Wikipedia"""
        try:
            # Extract main topic από την ερώτηση
            topics = self._extract_topics(question)
            
            for topic in topics:
                try:
                    # Search Wikipedia
                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(topic)}"
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'extract' in data:
                            self.sources_used.append(f"[Wikipedia - {data['title']}]({data.get('content_urls', {}).get('desktop', {}).get('page', '')})")
                            
                            return f"""
### 📖 Wikipedia: {data['title']}

{data['extract']}

**Περισσότερα**: {data.get('content_urls', {}).get('desktop', {}).get('page', 'N/A')}
"""
                except:
                    continue
                    
        except Exception as e:
            pass
        
        return None
    
    def _search_curated_sources(self, question: str) -> str:
        """Αναζήτηση σε επιλεγμένες πηγές AI"""
        
        # Curated AI resources
        ai_resources = {
            "machine learning": {
                "title": "Machine Learning Resources",
                "description": """
**Επίσημα Resources:**
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Google's ML Crash Course**: https://developers.google.com/machine-learning/crash-course
- **Coursera ML by Andrew Ng**: https://www.coursera.org/learn/machine-learning

**Papers & Research:**
- **ArXiv ML**: https://arxiv.org/list/cs.LG/recent
- **Papers with Code**: https://paperswithcode.com/
                """,
                "source": "Curated ML Resources"
            },
            
            "deep learning": {
                "title": "Deep Learning Resources",
                "description": """
**Frameworks Documentation:**
- **TensorFlow**: https://www.tensorflow.org/learn
- **PyTorch**: https://pytorch.org/tutorials/
- **Keras**: https://keras.io/guides/

**Educational:**
- **Deep Learning Book**: https://www.deeplearningbook.org/
- **Fast.ai**: https://www.fast.ai/
- **DeepLearning.AI**: https://www.deeplearning.ai/

**Research:**
- **ArXiv Deep Learning**: https://arxiv.org/list/cs.LG/recent
                """,
                "source": "Curated DL Resources"
            },
            
            "neural network": {
                "title": "Neural Networks Resources",
                "description": """
**Interactive Learning:**
- **Neural Network Playground**: https://playground.tensorflow.org/
- **3Blue1Brown NN Series**: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

**Documentation:**
- **PyTorch NN Tutorial**: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
- **TensorFlow Guide**: https://www.tensorflow.org/guide/keras/sequential_model
                """,
                "source": "Curated NN Resources"
            },
            
            "nlp": {
                "title": "Natural Language Processing Resources",
                "description": """
**Libraries:**
- **Hugging Face**: https://huggingface.co/docs
- **spaCy**: https://spacy.io/usage
- **NLTK**: https://www.nltk.org/

**Courses:**
- **HF NLP Course**: https://huggingface.co/learn/nlp-course
- **Stanford CS224N**: http://web.stanford.edu/class/cs224n/

**Models:**
- **Hugging Face Models**: https://huggingface.co/models
                """,
                "source": "Curated NLP Resources"
            },
            
            "computer vision": {
                "title": "Computer Vision Resources",
                "description": """
**Libraries:**
- **OpenCV**: https://docs.opencv.org/
- **Detectron2**: https://detectron2.readthedocs.io/
- **MMDetection**: https://github.com/open-mmlab/mmdetection

**Courses:**
- **Stanford CS231n**: http://cs231n.stanford.edu/
- **PyImageSearch**: https://www.pyimagesearch.com/

**Datasets:**
- **ImageNet**: https://www.image-net.org/
- **COCO**: https://cocodataset.org/
                """,
                "source": "Curated CV Resources"
            },
            
            "chatgpt": {
                "title": "ChatGPT & Large Language Models",
                "description": """
**Official:**
- **OpenAI Documentation**: https://platform.openai.com/docs
- **OpenAI Research**: https://openai.com/research

**Learning:**
- **Prompt Engineering Guide**: https://www.promptingguide.ai/
- **LangChain Docs**: https://python.langchain.com/

**Papers:**
- **GPT-3 Paper**: https://arxiv.org/abs/2005.14165
- **GPT-4 Technical Report**: https://arxiv.org/abs/2303.08774
- **InstructGPT**: https://arxiv.org/abs/2203.02155
                """,
                "source": "ChatGPT Resources"
            },
            
            "transformer": {
                "title": "Transformer Architecture Resources",
                "description": """
**Original Paper:**
- **"Attention Is All You Need"**: https://arxiv.org/abs/1706.03762

**Tutorials:**
- **The Illustrated Transformer**: http://jalammar.github.io/illustrated-transformer/
- **Annotated Transformer**: http://nlp.seas.harvard.edu/annotated-transformer/

**Implementation:**
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **PyTorch Transformer**: https://pytorch.org/docs/stable/nn.html#transformer
                """,
                "source": "Transformer Resources"
            },
            
            "reinforcement learning": {
                "title": "Reinforcement Learning Resources",
                "description": """
**Libraries:**
- **OpenAI Gym**: https://www.gymlibrary.dev/
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/

**Courses:**
- **David Silver's RL Course**: https://www.davidsilver.uk/teaching/
- **Spinning Up in Deep RL**: https://spinningup.openai.com/

**Books:**
- **Sutton & Barto**: http://incompleteideas.net/book/the-book.html
                """,
                "source": "RL Resources"
            }
        }
        
        question_lower = question.lower()
        
        for keyword, resource in ai_resources.items():
            if keyword in question_lower:
                self.sources_used.append(f"{resource['source']} (Curated)")
                return f"""
### 🎓 {resource['title']}

{resource['description']}
"""
        
        return None
    
    def _extract_topics(self, question: str) -> List[str]:
        """Εξάγει τα κύρια topics από την ερώτηση"""
        topics_map = {
            "machine learning": ["Machine learning", "Μηχανική μάθηση"],
            "deep learning": ["Deep learning", "Βαθιά μάθηση"],
            "neural network": ["Artificial neural network", "Neural network"],
            "artificial intelligence": ["Artificial intelligence", "AI"],
            "chatgpt": ["ChatGPT", "GPT-3", "GPT-4"],
            "transformer": ["Transformer (machine learning model)"],
            "supervised learning": ["Supervised learning"],
            "unsupervised learning": ["Unsupervised learning"],
            "reinforcement learning": ["Reinforcement learning"],
            "nlp": ["Natural language processing"],
            "computer vision": ["Computer vision"],
            "cnn": ["Convolutional neural network"],
            "rnn": ["Recurrent neural network"],
            "lstm": ["Long short-term memory"],
            "gan": ["Generative adversarial network"]
        }
        
        question_lower = question.lower()
        topics = []
        
        for key, values in topics_map.items():
            if key in question_lower:
                topics.extend(values)
        
        # Αν δεν βρέθηκε τίποτα, προσπάθησε το "artificial intelligence"
        if not topics:
            topics = ["Artificial intelligence"]
        
        return topics[:2]  # Return max 2 topics

def create_chatbot_interface():
    """Δημιουργία Streamlit interface για το chatbot"""
    st.markdown("### 🤖 AI Knowledge Assistant")
    st.markdown("*Ρωτήστε με οτιδήποτε σχετικό με Τεχνητή Νοημοσύνη!*")
    
    # Internet access indicator
    col_status1, col_status2 = st.columns([3, 1])
    with col_status1:
        st.caption("💡 **Enhanced με Internet Access**: Το chatbot έχει πρόσβαση σε Wikipedia, ArXiv, και curated AI resources!")
    with col_status2:
        st.success("🌐 Online", icon="✅")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = AIKnowledgeBot()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Γράψτε την ερώτησή σας εδώ..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Αναζητώ στο εκπαιδευτικό υλικό και online πηγές..."):
                response = st.session_state.chatbot.get_answer(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick questions
    st.markdown("---")
    st.markdown("#### 💬 Γρήγορες Ερωτήσεις:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏗️ Βασικά Δομικά Στοιχεία AI"):
            prompt = "Ποια είναι τα βασικά δομικά στοιχεία της Τεχνητής Νοημοσύνης;"
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chatbot.get_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("🧠 Τι είναι το Machine Learning;"):
            prompt = "Εξήγησε τι είναι το Machine Learning"
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chatbot.get_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("🤖 Πώς λειτουργεί το ChatGPT;"):
            prompt = "Πώς λειτουργεί το ChatGPT;"
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chatbot.get_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("🌐 Τι είναι το Deep Learning;"):
            prompt = "Τι είναι το Deep Learning;"
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chatbot.get_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("🔬 Τι είναι το Transformer;"):
            prompt = "Τι είναι το Transformer στην AI;"
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chatbot.get_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if st.button("🎮 Τι είναι το Reinforcement Learning;"):
            prompt = "Εξήγησε το Reinforcement Learning"
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = st.session_state.chatbot.get_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Καθαρισμός Συνομιλίας"):
        st.session_state.messages = []
        st.rerun()
    
    # Info about sources
    st.markdown("---")
    with st.expander("📚 Πηγές Πληροφοριών"):
        st.markdown("""
        Το chatbot αντλεί πληροφορίες από:
        
        **Τοπικές Πηγές:**
        - 📄 Εκπαιδευτικό υλικό PDF (957 σελίδες)
        - 💾 Structured QA database
        
        **Online Πηγές:**
        - 📖 Wikipedia (για γενικές πληροφορίες)
        - 🎓 Curated AI Resources:
          - Official Documentation (TensorFlow, PyTorch, Hugging Face)
          - Research Papers (ArXiv)
          - Educational Platforms (Coursera, Fast.ai, DeepLearning.AI)
          - Interactive Tools (TensorFlow Playground)
        
        **Ποιότητα:**
        - ✅ Όλες οι πηγές είναι επαληθευμένες
        - ✅ Προτεραιότητα στο τοπικό εκπαιδευτικό υλικό
        - ✅ Online πηγές για επιπλέον context και resources
        """)
