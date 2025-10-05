# AI Knowledge Base Chatbot Module - Simplified Working Version
# Πλήρως λειτουργική έκδοση χωρίς εξαρτήσεις

import streamlit as st

class AIKnowledgeBot:
    """
    Απλοποιημένο αλλά πλήρως λειτουργικό chatbot για AI εκπαίδευση
    """
    
    def __init__(self):
        self.qa_database = self._create_qa_database()
        
    def _create_qa_database(self):
        """Δημιουργία βάσης γνώσης με απαντήσεις"""
        return {
            "ai_definition": {
                "keywords": ["τεχνητή νοημοσύνη", "ορισμός", "τι είναι", "ai", "artificial intelligence", "νοημοσύνη", "define", "περίγραψε"],
                "answer": """
## 🤖 Τι είναι η Τεχνητή Νοημοσύνη;

### 📖 Ορισμός

Η **Τεχνητή Νοημοσύνη (Artificial Intelligence - AI)** είναι:

> "Ο κλάδος της επιστήμης των υπολογιστών που ασχολείται με τη δημιουργία ευφυών συστημάτων - μηχανών που μπορούν να σκέφτονται, να μαθαίνουν και να αποφασίζουν σαν (ή καλύτερα από) ανθρώπους!"

### 🎯 Βασικοί Στόχοι της AI

1. **Μάθηση** 🧠 - Απόκτηση γνώσης από δεδομένα
2. **Επίλυση Προβλημάτων** 🧩 - Αυτοματοποίηση αποφάσεων
3. **Αναγνώριση Προτύπων** 🔍 - Εύρεση patterns σε δεδομένα
4. **Λήψη Αποφάσεων** 🎯 - Έξυπνες, τεκμηριωμένες επιλογές
5. **Αυτοματοποίηση** 🤖 - Εκτέλεση εργασιών χωρίς ανθρώπινη παρέμβαση

### 💼 Εφαρμογές στην Καθημερινότητα

- 📱 **Smartphones** (Siri, Google Assistant)
- 🎬 **Streaming** (Netflix recommendations)
- 🚗 **Autonomous Vehicles**
- 🏥 **Healthcare** (διαγνώσεις)
- 💰 **Finance** (fraud detection)
- 🎓 **Education** (personalized learning)

### 🎭 Τύποι AI

**Narrow AI** (Εξειδικευμένη) - ✅ Υπάρχει σήμερα
- AlphaGo, Tesla Autopilot, Siri

**General AI** (Γενική) - ⏳ Δεν υπάρχει ακόμα
- Θα μπορεί να κάνει ΟΠΟΙΑΔΗΠΟΤΕ νοητική εργασία

**Super AI** (Υπερνοημοσύνη) - 🔮 Θεωρητικό
- Θα ξεπερνά ανθρώπινη νοημοσύνη σε όλα
"""
            },
            
            "building_blocks": {
                "keywords": ["δομικά", "στοιχεία", "βάση", "θεμέλιο", "components", "building blocks", "βασικά"],
                "answer": """
## 🏗️ Βασικά Δομικά Στοιχεία της AI

### 1. 📊 Δεδομένα (Data)

**Η βάση κάθε AI συστήματος**

- **Τύποι**: Κείμενο, εικόνες, ήχος, αριθμοί, βίντεο
- **Ποιότητα**: Ακρίβεια, πληρότητα, συνέπεια
- **Ποσότητα**: Όσο περισσότερα, τόσο καλύτερα (συνήθως)

**Παραδείγματα:**
- Netflix: 100M+ χρήστες, δισεκατομμύρια interactions
- Tesla: Εκατομμύρια miles από αυτόνομη οδήγηση

### 2. ⚙️ Αλγόριθμοι (Algorithms)

**Οι "συνταγές" που επεξεργάζονται τα δεδομένα**

**Κατηγορίες:**
- **Supervised Learning**: Logistic Regression, SVM, Random Forest, Neural Networks
- **Unsupervised Learning**: K-Means, PCA, Hierarchical Clustering
- **Reinforcement Learning**: Q-Learning, PPO, DQN

### 3. 🎯 Μοντέλα (Models)

**Το αποτέλεσμα της εκπαίδευσης**

**Lifecycle:**
1. Training (εκπαίδευση)
2. Evaluation (αξιολόγηση)
3. Deployment (παραγωγή)
4. Monitoring (παρακολούθηση)

**Παραδείγματα:**
- GPT-4: 175B+ parameters
- YOLOv8: Real-time object detection
- AlphaFold: Protein structure prediction

### 4. 💻 Υποδομές (Infrastructure)

**Hardware:**
- **CPUs**: General-purpose computing
- **GPUs**: NVIDIA A100, V100 (10-100x ταχύτερα)
- **TPUs**: Google's custom AI chips
- **NPUs**: Mobile devices (Apple Neural Engine)

**Software:**
- **Frameworks**: TensorFlow, PyTorch, Keras
- **Cloud**: AWS SageMaker, Google Cloud AI, Azure ML
- **Tools**: Docker, Kubernetes, MLflow

**Κόστος:**
- GPT-3 Training: ~$4.6M
- BERT-base: ~$7K
- Google Colab: FREE GPU! 🎉
"""
            },
            
            "machine_learning": {
                "keywords": ["machine learning", "μηχανική μάθηση", "ml", "μάθηση", "τύποι"],
                "answer": """
## 🧠 Machine Learning - Μηχανική Μάθηση

### 📖 Τι είναι;

> "Η ικανότητα των υπολογιστών να μαθαίνουν από δεδομένα χωρίς να προγραμματίζονται ρητά για κάθε εργασία!"

### 🎯 Τρεις Κύριοι Τύποι

#### 1️⃣ **Supervised Learning** (Επιβλεπόμενη Μάθηση)

**Τι είναι:**
- Εκπαίδευση με **labeled data** (δεδομένα με ετικέτες)
- Το μοντέλο μαθαίνει τη σχέση input → output

**Τύποι:**
- **Classification**: Πρόβλεψη κατηγορίας (spam/not spam)
- **Regression**: Πρόβλεψη αριθμού (τιμή σπιτιού)

**Αλγόριθμοι:**
- Logistic Regression
- Decision Trees, Random Forest
- Support Vector Machines (SVM)
- Neural Networks

**Εφαρμογές:**
- Email spam detection
- Medical diagnosis
- Stock price prediction
- House price estimation

#### 2️⃣ **Unsupervised Learning** (Μη Επιβλεπόμενη Μάθηση)

**Τι είναι:**
- Μάθηση από **unlabeled data**
- Ανακάλυψη κρυφών patterns

**Τύποι:**
- **Clustering**: Ομαδοποίηση (K-Means)
- **Dimensionality Reduction**: Μείωση features (PCA)
- **Anomaly Detection**: Εύρεση outliers

**Εφαρμογές:**
- Customer segmentation
- Fraud detection
- Recommendation systems
- Data compression

#### 3️⃣ **Reinforcement Learning** (Ενισχυτική Μάθηση)

**Τι είναι:**
- Μάθηση μέσω **trial-and-error**
- Agent παίρνει **rewards/penalties**

**Concepts:**
- Agent: Το σύστημα που μαθαίνει
- Environment: Ο κόσμος που αλληλεπιδρά
- Actions: Ενέργειες του agent
- Rewards: Ανταμοιβές (+ ή -)

**Αλγόριθμοι:**
- Q-Learning
- Deep Q-Networks (DQN)
- PPO, A3C

**Εφαρμογές:**
- AlphaGo (παίζει Go)
- Robotic control
- Autonomous vehicles
- Game AI

### 📊 ML Pipeline

1. **Data Collection** → Συλλογή δεδομένων
2. **Data Preprocessing** → Καθαρισμός
3. **Model Selection** → Επιλογή αλγορίθμου
4. **Training** → Εκπαίδευση
5. **Evaluation** → Αξιολόγηση
6. **Deployment** → Παραγωγή

### 🔄 Training Process

```
Input Data → Model → Predictions
              ↑
        Adjust weights
              ↑
        Calculate error
```

### 📚 Βιβλιοθήκες Python

- **scikit-learn**: ML γενικού σκοπού
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast ML
- **CatBoost**: Categorical features
"""
            },
            
            "deep_learning": {
                "keywords": ["deep learning", "βαθιά μάθηση", "neural network", "νευρωνικό", "layers"],
                "answer": """
## 🌊 Deep Learning - Βαθιά Μάθηση

### 📖 Τι είναι;

> "Νευρωνικά δίκτυα με **πολλά layers** (βάθος) που μαθαίνουν πολύπλοκες αναπαραστάσεις από δεδομένα!"

### 🧬 Neural Networks - Βασικά

**Δομή:**
- **Input Layer**: Εισαγωγή δεδομένων
- **Hidden Layers**: Επεξεργασία (το "βάθος")
- **Output Layer**: Έξοδος/Πρόβλεψη

**Neurons (Νευρώνες):**
- Λαμβάνουν inputs
- Πολλαπλασιάζουν με **weights** (βάρη)
- Προσθέτουν **bias**
- Εφαρμόζουν **activation function** (ReLU, Sigmoid)
- Παράγουν output

### 🏗️ Αρχιτεκτονικές

#### 1. **CNN** (Convolutional Neural Networks)

**Για τι:**
- Εικόνες και video
- Computer Vision

**Layers:**
- Convolutional layers (feature extraction)
- Pooling layers (downsampling)
- Fully connected layers (classification)

**Εφαρμογές:**
- Image classification
- Object detection (YOLO, R-CNN)
- Face recognition
- Self-driving cars

**Famous Models:**
- ResNet, VGG, Inception
- EfficientNet

#### 2. **RNN** (Recurrent Neural Networks)

**Για τι:**
- Sequential data
- Κείμενο, χρονοσειρές

**Τύποι:**
- **LSTM** (Long Short-Term Memory)
- **GRU** (Gated Recurrent Unit)

**Εφαρμογές:**
- Machine translation
- Speech recognition
- Time series prediction
- Text generation

#### 3. **Transformers**

**Επανάσταση στο NLP!**

**Key Innovation:**
- **Self-attention mechanism**
- Parallel processing (γρηγορότερο)
- Better long-range dependencies

**Famous Models:**
- **BERT**: Bidirectional Encoder
- **GPT**: Generative Pre-trained Transformer
- **T5**: Text-to-text
- **Vision Transformer (ViT)**: για εικόνες

### ⚡ Activation Functions

- **ReLU**: f(x) = max(0, x) - Πιο δημοφιλής
- **Sigmoid**: f(x) = 1/(1+e^-x) - Binary classification
- **Tanh**: f(x) = (e^x - e^-x)/(e^x + e^-x)
- **Softmax**: Multi-class classification

### 🔄 Training Process

**Backpropagation:**
1. Forward pass (υπολογισμός πρόβλεψης)
2. Calculate loss (σφάλμα)
3. Backward pass (gradient descent)
4. Update weights
5. Repeat!

**Optimizers:**
- SGD (Stochastic Gradient Descent)
- **Adam** (Adaptive Moment Estimation) ← Πιο δημοφιλής
- RMSprop
- AdaGrad

### 📊 Regularization Techniques

**Πρόβλημα: Overfitting**
- Το μοντέλο "μαθαίνει" το training set απ' έξω

**Λύσεις:**
- **Dropout**: Απενεργοποίηση τυχαίων neurons
- **L1/L2 Regularization**: Ποινή στα μεγάλα weights
- **Batch Normalization**: Normalization μεταξύ layers
- **Early Stopping**: Σταμάτα όταν validation loss αυξάνεται

### 💻 Frameworks

- **TensorFlow**: Google (production-ready)
- **PyTorch**: Facebook (research favorite)
- **Keras**: High-level API (user-friendly)
- **JAX**: High-performance

### 🚀 Εφαρμογές

**Computer Vision:**
- Medical imaging
- Autonomous vehicles
- Facial recognition

**NLP:**
- ChatGPT, Google Translate
- Sentiment analysis
- Question answering

**Other:**
- Drug discovery (AlphaFold)
- Speech recognition (Alexa, Siri)
- Game AI (AlphaGo, OpenAI Five)
"""
            },
            
            "chatgpt": {
                "keywords": ["chatgpt", "gpt", "language model", "γλωσσικό", "openai", "llm"],
                "answer": """
## 🤖 ChatGPT - Πώς Λειτουργεί;

### 📖 Τι είναι;

> "Ένα **Large Language Model** (LLM) της OpenAI που μπορεί να κατανοεί και να παράγει ανθρώπινη γλώσσα!"

### 🏗️ Αρχιτεκτονική

**Βάση: Transformer**
- Εισήχθη το 2017 ("Attention Is All You Need")
- **Self-attention mechanism**
- Parallel processing

**GPT = Generative Pre-trained Transformer**

### 🔄 Πώς Λειτουργεί;

#### Βήμα 1: **Tokenization**
```
Input: "Hello world"
→ Tokens: ["Hello", " world"]
```

#### Βήμα 2: **Understanding**
- Αναγνωρίζει γραμματική, συντακτικό
- Κατανοεί πλαίσιο (context)
- Αναλύει νόημα

#### Βήμα 3: **Generation**
- Προβλέπει την **επόμενη λέξη**
- Επαναλαμβάνει για να δημιουργήσει κείμενο
- Διατηρεί συνοχή

#### Βήμα 4: **Response**
```
Output: Συνεκτική απάντηση!
```

### 📚 Εκπαίδευση

**Phase 1: Pre-training**
- Τεράστια datasets (βιβλία, Wikipedia, κώδικας)
- Unsupervised learning
- Μαθαίνει γλώσσα γενικά

**Phase 2: Fine-tuning**
- Supervised learning με human feedback
- **RLHF** (Reinforcement Learning from Human Feedback)
- Μαθαίνει να είναι helpful, truthful, harmless

### 🎯 Δυνατότητες

✅ **Δημιουργία κειμένου**
- Άρθρα, ποιήματα, ιστορίες

✅ **Συνομιλία**
- Natural dialogue

✅ **Σύνοψη**
- TL;DR long texts

✅ **Μετάφραση**
- 100+ γλώσσες

✅ **Προγραμματισμός**
- Code generation & debugging

✅ **Ανάλυση**
- Sentiment, intent, entities

### ⚠️ Περιορισμοί

❌ **Hallucinations**
- Μπορεί να "εφευρίσκει" πληροφορίες

❌ **Knowledge Cutoff**
- Δεν έχει πρόσβαση σε real-time info (base model)

❌ **No True Understanding**
- Pattern matching, όχι πραγματική κατανόηση

❌ **Bias**
- Αναπαράγει προκαταλήψεις από training data

### 🔮 Εξελίξεις

**GPT-3.5** (ChatGPT launch):
- 175B parameters

**GPT-4**:
- Multimodal (text + images)
- Longer context window
- More capable

**Μέλλον:**
- GPT-5?
- More multimodal
- Better reasoning
- Real-world actions

### 💡 Best Practices

**Prompt Engineering:**
- Να είστε συγκεκριμένοι
- Δώστε παραδείγματα
- Ζητήστε να "σκεφτεί step-by-step"
- Ορίστε το role ("You are an expert...")

**Παράδειγμα Καλού Prompt:**
```
"Εξήγησε την αρχιτεκτονική Transformer σε ένα 
10-χρονο παιδί, χρησιμοποιώντας απλές αναλογίες 
και παραδείγματα από την καθημερινή ζωή."
```
"""
            },
            
            "applications": {
                "keywords": ["εφαρμογές", "applications", "χρήσεις", "uses", "τομείς"],
                "answer": """
## 💼 Εφαρμογές της AI - Πού Χρησιμοποιείται;

### 🏥 Υγεία (Healthcare)

**Διάγνωση:**
- Ανάλυση ιατρικών εικόνων (X-rays, CT, MRI)
- Ανίχνευση καρκίνου
- Πρόβλεψη ασθενειών

**Ανακάλυψη Φαρμάκων:**
- AlphaFold: Protein folding
- Drug design με AI
- Clinical trials optimization

**Προσωποποιημένη Ιατρική:**
- Tailored treatments
- Genomics analysis

**Παραδείγματα:**
- IBM Watson Health
- Google DeepMind Health
- PathAI (παθολογία)

### 🚗 Μεταφορές (Transportation)

**Autonomous Vehicles:**
- Tesla Autopilot
- Waymo (Google)
- Cruise (GM)

**Τεχνολογίες:**
- Computer Vision (cameras)
- LiDAR, Radar
- Path planning
- Object detection

**Επίπεδα Αυτονομίας:**
- Level 0: Καμία
- Level 2: Partial (Tesla)
- Level 5: Full autonomy

### 💰 Χρηματοοικονομικά (Finance)

**Ανίχνευση Απάτης:**
- Real-time fraud detection
- Anomaly detection

**Trading:**
- Algorithmic trading
- Market prediction

**Risk Management:**
- Credit scoring
- Loan approval

**Robo-Advisors:**
- Automated investment advice

### 🎓 Εκπαίδευση (Education)

**Personalized Learning:**
- Adaptive content
- Individual pace

**Intelligent Tutoring Systems:**
- 24/7 assistance
- Immediate feedback

**Automated Grading:**
- Essays, code, math

**Content Creation:**
- Quiz generation
- Study materials

### 🏪 Πωλήσεις & Marketing

**Recommendations:**
- Netflix: "You might like..."
- Amazon: "Customers also bought..."

**Chatbots:**
- Customer service 24/7
- Lead generation

**Predictive Analytics:**
- Customer churn
- Sales forecasting

**Ad Targeting:**
- Personalized ads
- Optimization

### 🏭 Βιομηχανία (Manufacturing)

**Quality Control:**
- Visual inspection
- Defect detection

**Predictive Maintenance:**
- Πρόβλεψη βλαβών
- Downtime reduction

**Robotics:**
- Assembly lines
- Warehouse automation

### 🎨 Δημιουργικότητα (Creativity)

**Image Generation:**
- DALL-E, Midjourney, Stable Diffusion
- Art creation

**Music:**
- AI composers
- Style transfer

**Writing:**
- Content creation
- Copywriting

**Video:**
- Deepfakes
- Video editing

### 🌐 Άλλοι Τομείς

**Agriculture:**
- Crop monitoring
- Disease detection

**Legal:**
- Document review
- Contract analysis

**Cybersecurity:**
- Threat detection
- Incident response

**Climate:**
- Weather prediction
- Climate modeling

### 📊 Στατιστικά

**Market Size:**
- $136B in 2022
- Projected $1.8T by 2030

**Adoption:**
- 35% of companies use AI
- 77% exploring AI

**Impact:**
- 40% productivity increase
- $15.7T economic impact by 2030
"""
            },
            
            "ethics": {
                "keywords": ["ηθική", "ethics", "bias", "μεροληψία", "διαφάνεια", "ιδιωτικότητα", "privacy"],
                "answer": """
## ⚖️ Ηθικά Ζητήματα της AI

### ⚠️ Κύριες Προκλήσεις

#### 1. **Bias & Fairness** (Μεροληψία & Δικαιοσύνη)

**Πρόβλημα:**
- AI αναπαράγει προκαταλήψεις από δεδομένα
- Διακρίσεις σε φυλή, φύλο, ηλικία

**Παραδείγματα:**
- Amazon hiring tool (bias κατά γυναικών)
- COMPAS (criminal justice bias)
- Facial recognition (λιγότερο ακριβής για POC)

**Λύσεις:**
- Diverse datasets
- Bias detection tools
- Fair AI algorithms
- Regular audits

#### 2. **Privacy & Security** (Ιδιωτικότητα)

**Πρόβλημα:**
- Συλλογή τεράστιων προσωπικών δεδομένων
- Data breaches
- Surveillance

**Concerns:**
- Ποιος έχει πρόσβαση;
- Πού αποθηκεύονται;
- Πώς χρησιμοποιούνται;

**Λύσεις:**
- **GDPR** compliance (EU)
- Data minimization
- Encryption
- Privacy-preserving AI
- Differential privacy

#### 3. **Transparency & Explainability** (Διαφάνεια)

**Πρόβλημα:**
- "Black box" models
- Δύσκολο να εξηγηθούν αποφάσεις

**Γιατί σημαντικό:**
- Trust
- Accountability
- Debugging
- Compliance

**Λύσεις:**
- **XAI** (Explainable AI)
- LIME, SHAP (interpretation tools)
- Simpler models όπου δυνατόν
- Documentation

#### 4. **Job Displacement** (Αντικατάσταση Εργασιών)

**Πρόβλημα:**
- Automation → Job loss
- Ανισότητες

**Επηρεαζόμενοι Τομείς:**
- Manufacturing
- Transportation (drivers)
- Customer service
- Data entry

**Αντιμετώπιση:**
- **Reskilling programs**
- Lifelong learning
- Universal Basic Income? (συζήτηση)
- New job creation
- Human-AI collaboration

#### 5. **Accountability** (Ευθύνη)

**Ερώτημα:**
- Ποιος ευθύνεται για λάθη AI;
- Developer? Company? User?

**Scenarios:**
- Αυτόνομο όχημα ατύχημα
- Λάθος ιατρική διάγνωση
- Algorithmic discrimination

**Λύσεις:**
- Clear regulations
- Liability frameworks
- AI governance
- Insurance models

#### 6. **Safety & Control** (Ασφάλεια)

**Concerns:**
- Autonomous weapons
- AGI risk (existential)
- Loss of control
- Misuse

**AI Safety Research:**
- Alignment problem
- Value learning
- Robustness
- Interpretability

### 📜 Πλαίσια & Κατευθυντήριες Γραμμές

#### **EU AI Act**
- Risk-based approach
- High-risk systems regulation
- Fines έως €30M

#### **IEEE Ethics Guidelines**
- Human Rights
- Well-being
- Accountability
- Transparency

#### **Partnership on AI**
- Multi-stakeholder initiative
- Best practices
- Research

### ✅ Αρχές Responsible AI

1. **Fairness** - Δίκαιες αποφάσεις
2. **Transparency** - Διαφάνεια
3. **Privacy** - Προστασία δεδομένων
4. **Safety** - Ασφάλεια
5. **Accountability** - Ευθύνη
6. **Human-Centric** - Ανθρωποκεντρικό

### 🔮 Μέλλον

**Προκλήσεις:**
- AGI alignment
- Global cooperation
- Equitable access
- Environmental impact (energy)

**Ευκαιρίες:**
- Solving grand challenges
- Democratization
- Human augmentation
- Scientific breakthroughs

### 💡 Τι Μπορείτε να Κάνετε;

**Ως Developers:**
- Consider ethics early
- Diverse teams
- Test for bias
- Document decisions

**Ως Users:**
- Be informed
- Ask questions
- Demand transparency
- Report issues

**Ως Πολίτες:**
- Support good regulation
- Education
- Public discourse
"""
            }
        }
    
    def get_answer(self, question: str) -> str:
        """Βρίσκει την καλύτερη απάντηση για την ερώτηση"""
        question_lower = question.lower()
        
        best_match = None
        max_score = 0
        
        # Αναζήτηση στη βάση
        for topic_key, topic_data in self.qa_database.items():
            score = sum(1 for keyword in topic_data["keywords"] if keyword in question_lower)
            if score > max_score:
                max_score = score
                best_match = topic_data["answer"]
        
        if best_match and max_score > 0:
            return best_match
        else:
            return self._generate_default_answer(question)
    
    def _generate_default_answer(self, question: str) -> str:
        """Προεπιλεγμένη απάντηση όταν δεν βρεθεί match"""
        return f"""
## 🤔 Δεν βρήκα συγκεκριμένη απάντηση

Η ερώτησή σας: **"{question}"**

### 💡 Προτάσεις - Μπορείτε να ρωτήσετε για:

📚 **Βασικές Έννοιες:**
- "Τι είναι η Τεχνητή Νοημοσύνη;"
- "Ποια είναι τα βασικά δομικά στοιχεία της AI;"

🧠 **Machine & Deep Learning:**
- "Εξήγησε το Machine Learning"
- "Τι είναι το Deep Learning;"
- "Διαφορά CNN και RNN"

🤖 **ChatGPT & LLMs:**
- "Πώς λειτουργεί το ChatGPT;"
- "Τι είναι το Transformer;"

💼 **Εφαρμογές:**
- "Πού χρησιμοποιείται η AI;"
- "AI στην υγεία"
- "Εφαρμογές στην εκπαίδευση"

⚖️ **Ηθική:**
- "Ηθικά ζητήματα της AI"
- "Bias στην AI"
- "Privacy και AI"

### 🎯 Tips για Καλύτερες Απαντήσεις:

✅ **Συγκεκριμένες ερωτήσεις** είναι καλύτερες
❌ "Πες μου για AI" → ⚠️ Πολύ γενικό
✅ "Εξήγησε τους τύπους Machine Learning" → 👍 Συγκεκριμένο

### 📖 Θέματα που Καλύπτω:

1. **Θεωρία**: Ορισμοί, concepts, ιστορία
2. **Τεχνολογία**: ML, DL, NLP, Computer Vision
3. **Πρακτική**: Εφαρμογές, frameworks, tools
4. **Ηθική**: Bias, privacy, accountability
"""


def create_chatbot_interface():
    """Δημιουργία Streamlit interface για το chatbot"""
    st.markdown("### 🤖 AI Knowledge Assistant")
    st.markdown("*Ρωτήστε με οτιδήποτε σχετικό με Τεχνητή Νοημοσύνη!*")
    
    # Initialize chatbot
    if 'bot' not in st.session_state:
        st.session_state.bot = AIKnowledgeBot()
    
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
                response = st.session_state.bot.get_answer(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Quick questions
    st.markdown("---")
    st.markdown("#### 💬 Γρήγορες Ερωτήσεις:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        ("🤖 Ορισμός AI", "Δώσε μου ένα ορισμό για την Τεχνητή Νοημοσύνη"),
        ("🏗️ Δομικά Στοιχεία", "Ποια είναι τα βασικά δομικά στοιχεία της AI;"),
        ("🧠 Machine Learning", "Εξήγησε το Machine Learning"),
        ("🌊 Deep Learning", "Τι είναι το Deep Learning;"),
        ("🤖 ChatGPT", "Πώς λειτουργεί το ChatGPT;"),
        ("💼 Εφαρμογές", "Πού χρησιμοποιείται η AI;"),
        ("⚖️ Ηθική", "Ποια είναι τα ηθικά ζητήματα της AI;"),
        ("🎯 Όλα", "Πες μου τα πάντα για AI")
    ]
    
    for i, (label, question) in enumerate(quick_questions):
        col = [col1, col2, col3, col4][i % 4]
        with col:
            if st.button(label, key=f"quick_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                response = st.session_state.bot.get_answer(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Clear button
    if st.button("🗑️ Καθαρισμός Συνομιλίας"):
        st.session_state.messages = []
        st.rerun()
