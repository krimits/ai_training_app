# AI Knowledge Base Chatbot Module

import re
from typing import List, Tuple, Dict
import streamlit as st

class AIKnowledgeBot:
    """
    Intelligent chatbot που απαντά σε ερωτήσεις βασισμένο στο εκπαιδευτικό υλικό AI.
    """
    
    def __init__(self, knowledge_file='pdf_content.txt'):
        self.knowledge_base = self._load_knowledge(knowledge_file)
        self.qa_pairs = self._create_qa_database()
        
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
        """Γενική απάντηση όταν δεν βρεθεί match"""
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

def create_chatbot_interface():
    """Δημιουργία Streamlit interface για το chatbot"""
    st.markdown("### 🤖 AI Knowledge Assistant")
    st.markdown("*Ρωτήστε με οτιδήποτε σχετικό με Τεχνητή Νοημοσύνη!*")
    
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
            with st.spinner("Σκέφτομαι..."):
                response = st.session_state.chatbot.get_answer(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick questions
    st.markdown("---")
    st.markdown("#### 💬 Γρήγορες Ερωτήσεις:")
    
    col1, col2 = st.columns(2)
    
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
    
    # Clear chat button
    if st.button("🗑️ Καθαρισμός Συνομιλίας"):
        st.session_state.messages = []
        st.rerun()
