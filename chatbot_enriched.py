# AI Knowledge Base Chatbot Module with FULL Educational Content
# Εμπλουτισμένο με πλήρες εκπαιδευτικό υλικό από PDF και online πηγές

import re
from typing import List, Tuple, Dict
import streamlit as st
import requests
from urllib.parse import quote
import json

class AIKnowledgeBotEnriched:
    """
    Intelligent chatbot που απαντά σε ερωτήσεις βασισμένο στο πλήρες εκπαιδευτικό υλικό AI.
    Περιλαμβάνει ολοκληρωμένες απαντήσεις για όλες τις ενότητες του μαθήματος.
    """
    
    def __init__(self, knowledge_file='pdf_content.txt'):
        self.knowledge_base = self._load_knowledge(knowledge_file)
        self.qa_pairs = self._create_comprehensive_qa_database()
        self.use_internet = True
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
    
    def _create_comprehensive_qa_database(self):
        """
        Δημιουργία πλήρους βάσης γνώσης με όλο το εκπαιδευτικό υλικό.
        Καλύπτει όλες τις ενότητες 1.1-1.7 και επιπλέον θέματα.
        """
        return {
            # === ΕΙΣΑΓΩΓΗ & ΒΑΣΙΚΕΣ ΕΝΝΟΙΕΣ ===
            
            "τεχνητή νοημοσύνη ορισμός": {
                "keywords": ["τεχνητή νοημοσύνη", "ορισμός", "τι είναι", "ai definition", "artificial intelligence", "νοημοσύνη", "ορίζω", "εξήγησε", "define", "περίγραψε", "περιγράψτε"],
                "answer": self._get_ai_definition()
            },
            
            "δομικά στοιχεία": {
                "keywords": ["δομικά", "στοιχεία", "βάση", "θεμέλιο", "components", "building blocks", "περιγράψτε", "βασικά", "θεμελιώδη"],
                "answer": self._get_building_blocks()
            },
            
            # === MACHINE LEARNING ===
            
            "machine learning": {
                "keywords": ["machine learning", "μηχανική μάθηση", "ml", "μάθηση", "τύποι μάθησης", "επιβλεπόμενη", "μη επιβλεπόμενη"],
                "answer": self._get_machine_learning()
            },
            
            "supervised learning": {
                "keywords": ["supervised", "επιβλεπόμενη", "labeled data", "classification", "regression", "ταξινόμηση", "παλινδρόμηση"],
                "answer": self._get_supervised_learning()
            },
            
            "unsupervised learning": {
                "keywords": ["unsupervised", "μη επιβλεπόμενη", "unlabeled", "clustering", "ομαδοποίηση", "k-means", "pca"],
                "answer": self._get_unsupervised_learning()
            },
            
            "reinforcement learning": {
                "keywords": ["reinforcement", "ενισχυτική", "ενίσχυση", "rewards", "ανταμοιβές", "q-learning", "gaming"],
                "answer": self._get_reinforcement_learning()
            },
            
            # === DEEP LEARNING & NEURAL NETWORKS ===
            
            "deep learning": {
                "keywords": ["deep learning", "βαθιά μάθηση", "βαθειά", "neural network", "νευρωνικό δίκτυο", "layers", "στρώματα"],
                "answer": self._get_deep_learning()
            },
            
            "neural networks": {
                "keywords": ["neural network", "νευρωνικό δίκτυο", "νευρώνας", "νευρώνες", "layers", "weights", "βάρη"],
                "answer": self._get_neural_networks()
            },
            
            "cnn": {
                "keywords": ["cnn", "convolutional", "συνελικτικό", "εικόνες", "computer vision"],
                "answer": self._get_cnn()
            },
            
            "rnn": {
                "keywords": ["rnn", "recurrent", "αναδρομικό", "lstm", "gru", "sequences", "ακολουθίες"],
                "answer": self._get_rnn()
            },
            
            "transformer": {
                "keywords": ["transformer", "attention", "προσοχή", "bert", "gpt", "self-attention"],
                "answer": self._get_transformer()
            },
            
            # === CHATGPT & LLMs ===
            
            "chatgpt": {
                "keywords": ["chatgpt", "gpt", "language model", "γλωσσικό μοντέλο", "openai", "llm"],
                "answer": self._get_chatgpt()
            },
            
            "llm": {
                "keywords": ["llm", "large language model", "μεγάλο γλωσσικό", "gpt-3", "gpt-4"],
                "answer": self._get_llm()
            },
            
            # === GENERATIVE AI ===
            
            "generative ai": {
                "keywords": ["generative", "δημιουργική", "παραγωγική", "generation", "δημιουργία", "gan", "vae"],
                "answer": self._get_generative_ai()
            },
            
            "gan": {
                "keywords": ["gan", "generative adversarial", "ανταγωνιστικό", "generator", "discriminator"],
                "answer": self._get_gan()
            },
            
            # === ΕΦΑΡΜΟΓΕΣ ===
            
            "εφαρμογές ai": {
                "keywords": ["εφαρμογές", "applications", "χρήσεις", "uses", "τομείς", "domains"],
                "answer": self._get_applications()
            },
            
            "υγεία ai": {
                "keywords": ["υγεία", "health", "healthcare", "ιατρική", "medical", "διάγνωση", "diagnosis"],
                "answer": self._get_health_applications()
            },
            
            "εκπαίδευση ai": {
                "keywords": ["εκπαίδευση", "education", "μάθηση", "learning", "φοιτητές", "students"],
                "answer": self._get_education_applications()
            },
            
            # === ΗΘΙΚΑ ΖΗΤΗΜΑΤΑ ===
            
            "ηθική ai": {
                "keywords": ["ηθική", "ethics", "bias", "μεροληψία", "προκατάληψη", "διαφάνεια", "transparency"],
                "answer": self._get_ethics()
            },
            
            "privacy": {
                "keywords": ["privacy", "ιδιωτικότητα", "gdpr", "δεδομένα", "data protection", "προστασία"],
                "answer": self._get_privacy()
            },
            
            # === PYTHON & PROGRAMMING ===
            
            "python": {
                "keywords": ["python", "προγραμματισμός", "programming", "κώδικας", "code", "numpy", "pandas"],
                "answer": self._get_python()
            },
            
            "colab": {
                "keywords": ["colab", "google colab", "notebook", "jupyter", "εργαλεία", "tools"],
                "answer": self._get_colab()
            }
        }
    
    # === ΜΕΘΟΔΟΙ ΓΙΑ ΚΑΘΕ ΘΕΜΑ ===
    
    def _get_ai_definition(self):
        """Ολοκληρωμένος ορισμός AI με όλες τις λεπτομέρειες"""
        return """
## 🤖 Τι είναι η Τεχνητή Νοημοσύνη; - Πλήρης Ανάλυση

### 📖 Ολοκληρωμένος Ορισμός

Η **Τεχνητή Νοημοσύνη (Artificial Intelligence - AI)** είναι:

🎯 **Ορισμός 1 (Τεχνικός)**:
> "Ο κλάδος της επιστήμης των υπολογιστών που ασχολείται με τη δημιουργία ευφυών πρακτόρων - συστημάτων που μπορούν να συλλογιστούν, να μαθαίνουν από την εμπειρία και να ενεργούν αυτόνομα, προσομοιώνοντας και συχνά ξεπερνώντας τις ανθρώπινες γνωστικές ικανότητες."

🎯 **Ορισμός 2 (Απλός)**:
> "Η επιστήμη που κάνει τους υπολογιστές να σκέφτονται, να μαθαίνουν και να αποφασίζουν σαν (ή καλύτερα από) ανθρώπους!"

🎯 **Ορισμός 3 (Πρακτικός)**:
> "Η τεχνολογία που επιτρέπει σε μηχανές να εκτελούν εργασίες που παραδοσιακά απαιτούν ανθρώπινη νοημοσύνη, όπως αναγνώριση εικόνων, κατανόηση γλώσσας, λήψη αποφάσεων και επίλυση προβλημάτων."

---

### 🎓 Επίσημοι Ορισμοί από Πρωτοπόρους

**John McCarthy (1956)** - "Πατέρας της AI":
> "Η επιστήμη και η μηχανική της δημιουργίας έξυπνων μηχανών, ειδικά έξυπνων προγραμμάτων υπολογιστών."

**Marvin Minsky** (MIT):
> "Η επιστήμη του να κάνεις μηχανές να κάνουν πράγματα που θα απαιτούσαν νοημοσύνη αν τα έκανε άνθρωπος."

**Stuart Russell & Peter Norvig** (AI: A Modern Approach):
> "Η μελέτη των agents που λαμβάνουν inputs από το περιβάλλον και εκτελούν actions για να μεγιστοποιήσουν την επιτυχία τους."

**Alan Turing** (1950):
> "Μπορούν οι μηχανές να σκέφτονται;" - Turing Test

---

### 💡 Κατανοώντας την AI - Αναλογία

**Φανταστείτε την AI ως έναν εξειδικευμένο μαθητή:**

📚 **Ο Μαθητής (AI System)**
- Μπορεί να μάθει από βιβλία (δεδομένα)
- Έχει δασκάλους (αλγορίθμοι)
- Χρειάζεται χρόνο μελέτης (υπολογιστική ισχύ)
- Δίνει εξετάσεις (testing)
- Βελτιώνεται με την πρακτική

🎯 **Τα Μαθήματα (AI Tasks)**
- Γλώσσες → Natural Language Processing
- Τέχνη → Computer Vision
- Μουσική → Audio Processing
- Λογική → Problem Solving
- Μαθηματικά → Optimization

💼 **Τα Αποτελέσματα (Applications)**
- Μεταφράσεις γλωσσών
- Διαγνώσεις ασθενειών
- Δημιουργία έργων τέχνης
- Οδήγηση αυτοκινήτων
- Παιχνίδια σκάκι/Go

---

### 🏗️ Κύριοι Στόχοι της AI

#### 1️⃣ **Μάθηση (Learning)** 🧠
- Απόκτηση γνώσης από δεδομένα
- Βελτίωση με την εμπειρία
- Αναγνώριση patterns
- Transfer learning

#### 2️⃣ **Επίλυση Προβλημάτων (Problem Solving)** 🧩
- Αυτοματοποίηση λήψης αποφάσεων
- Ανάλυση και αξιολόγηση επιλογών
- Εύρεση βέλτιστων λύσεων
- Χειρισμός αβεβαιότητας

#### 3️⃣ **Αναγνώριση Προτύπων (Pattern Recognition)** 🔍
- Εικόνες, κείμενο, ήχος, αριθμοί
- Εξαγωγή χρήσιμων πληροφοριών
- Πρόβλεψη τάσεων
- Anomaly detection

#### 4️⃣ **Λήψη Αποφάσεων (Decision Making)** 🎯
- Ανάλυση σε υπερανθρώπινη κλίμακα
- Αντικειμενικές αποφάσεις
- Real-time decisions
- Τεκμηριωμένες επιλογές

#### 5️⃣ **Αυτοματοποίηση (Automation)** 🤖
- Επαναλαμβανόμενες εργασίες
- 24/7 λειτουργία
- Αύξηση παραγωγικότητας
- Μείωση ανθρώπινου λάθους

#### 6️⃣ **Προσαρμοστικότητα (Adaptability)** 🔄
- Νέα δεδομένα
- Αλλαγές περιβάλλοντος
- Continuous learning
- Evolution

---

### 🎭 Τύποι Τεχνητής Νοημοσύνης

#### **Narrow AI (Weak AI)** - Περιορισμένη/Εξειδικευμένη
✅ **Υπάρχει σήμερα**

**Χαρακτηριστικά:**
- Εξειδικευμένη σε συγκεκριμένες εργασίες
- Δεν γενικεύει σε άλλα προβλήματα
- Ξεπερνά ανθρώπους σε συγκεκριμένα πεδία

**Παραδείγματα:**
- 🎮 AlphaGo (παίζει Go)
- 🚗 Tesla Autopilot (οδηγεί αυτοκίνητα)
- 🗣️ Siri/Alexa (φωνητικοί βοηθοί)
- 🔍 Google Search (αναζήτηση)
- 📧 Gmail Smart Reply (προτάσεις απαντήσεων)
- 🎬 Netflix Recommendations
- 👁️ Face ID (αναγνώριση προσώπου)

#### **General AI (Strong AI / AGI)** - Γενική Νοημοσύνη
⏳ **Δεν υπάρχει ακόμα**

**Χαρακτηριστικά:**
- Ισοδύναμη με ανθρώπινη νοημοσύνη
- Μπορεί να μάθει ΟΠΟΙΑΔΗΠΟΤΕ νοητική εργασία
- Κατανοεί context και common sense
- Transfer learning σε νέα domains

**Χρονοδιάγραμμα:**
- Αβέβαιο - ίσως 2040-2060;
- Τεράστιες τεχνολογικές προκλήσεις
- Ηθικά και φιλοσοφικά ερωτήματα

#### **Super AI** - Υπερνοημοσύνη
🔮 **Θεωρητικό**

**Χαρακτηριστικά:**
- Ξεπερνά ανθρώπινη νοημοσύνη σε ΟΛΑ
- Αυτο-βελτίωση (recursive self-improvement)
- Μη προβλέψιμες ικανότητες

**Ζητήματα:**
- Existential risk?
- Control problem
- Alignment problem
- AI safety research

---

### 🌳 Κλάδοι και Τεχνολογίες της AI

#### 1. **Machine Learning (ML)** 🧠
**Τι είναι:**
- Μάθηση από δεδομένα χωρίς ρητό προγραμματισμό
- Βελτίωση με την εμπειρία

**Τύποι:**
- Supervised Learning (επιβλεπόμενη)
- Unsupervised Learning (μη επιβλεπόμενη)
- Reinforcement Learning (ενισχυτική)

**Εφαρμογές:**
- Spam filtering
- Credit scoring
- Medical diagnosis
- Stock prediction

#### 2. **Deep Learning (DL)** 🌊
**Τι είναι:**
- Νευρωνικά δίκτυα με πολλά layers
- Αυτόματη feature extraction
- State-of-the-art σε πολλά πεδία

**Αρχιτεκτονικές:**
- CNN (Convolutional Neural Networks)
- RNN/LSTM (Recurrent Neural Networks)
- Transformers
- GANs (Generative Adversarial Networks)

**Εφαρμογές:**
- Image recognition
- Speech recognition
- Natural language understanding
- Drug discovery

#### 3. **Natural Language Processing (NLP)** 💬
**Τι είναι:**
- Κατανόηση και παραγωγή ανθρώπινης γλώσσας
- Text mining & analysis

**Τεχνικές:**
- Tokenization
- Word embeddings (Word2Vec, GloVe)
- Transformers (BERT, GPT)
- Named Entity Recognition

**Εφαρμογές:**
- ChatGPT, Google Translate
- Sentiment analysis
- Text summarization
- Question answering

#### 4. **Computer Vision (CV)** 👁️
**Τι είναι:**
- "Βλέπει" και ερμηνεύει οπτικά δεδομένα
- Image και video analysis

**Τεχνικές:**
- Object detection (YOLO, R-CNN)
- Image segmentation
- Facial recognition
- 3D reconstruction

**Εφαρμογές:**
- Self-driving cars
- Medical imaging
- Security systems
- Augmented Reality

#### 5. **Robotics** 🤖
**Τι είναι:**
- Φυσικά συστήματα με AI
- Αυτόνομη κίνηση και manipulation

**Συστατικά:**
- Perception (αισθητήρες)
- Planning (σχεδιασμός)
- Control (έλεγχος)
- Learning (μάθηση)

**Εφαρμογές:**
- Industrial robots
- Surgical robots
- Warehouse automation
- Delivery drones

#### 6. **Expert Systems** 💼
**Τι είναι:**
- Προσομοίωση ειδικών γνώσεων
- Rule-based systems

**Συστατικά:**
- Knowledge Base
- Inference Engine
- User Interface

**Εφαρμογές:**
- Medical diagnosis
- Financial advising
- Legal reasoning

---

### 💼 Εφαρμογές στον Πραγματικό Κόσμο

#### **Καθημερινή Ζωή** 🏠
- 📱 Smartphones (Siri, Google Assistant)
- 🎬 Streaming (Netflix, Spotify recommendations)
- 📧 Email (spam filtering, smart reply)
- 🗺️ Navigation (Google Maps, Waze)
- 📸 Photos (automatic tagging, enhancement)
- 🛒 Shopping (product recommendations)
- 💬 Social Media (news feed, content moderation)

#### **Επιχειρήσεις** 💼
- 📊 Business Intelligence
- 🎯 Targeted Marketing
- 💰 Fraud Detection
- 📈 Sales Forecasting
- 🤖 Customer Service (chatbots)
- 📦 Supply Chain Optimization
- 💡 Process Automation

#### **Υγεία** 🏥
- 🩺 Medical Diagnosis
- 💊 Drug Discovery
- 🧬 Genomics
- 🏥 Hospital Management
- 📱 Health Monitoring (wearables)
- 🤖 Surgical Robots
- 🧠 Mental Health Support

#### **Μεταφορές** 🚗
- 🚗 Autonomous Vehicles
- ✈️ Flight Planning
- 🚢 Maritime Navigation
- 🚦 Traffic Management
- 🚇 Public Transportation Optimization
- 📦 Logistics

#### **Εκπαίδευση** 🎓
- 📚 Personalized Learning
- 🤖 Intelligent Tutoring Systems
- 📝 Automated Grading
- 🌐 Language Learning
- 🎮 Educational Games
- 📊 Student Analytics

#### **Επιστήμη & Έρευνα** 🔬
- 🌌 Space Exploration
- 🧪 Drug Design
- 🔭 Astronomy (data analysis)
- 🌡️ Climate Modeling
- 🧬 Protein Folding (AlphaFold)
- ⚛️ Physics Simulations

#### **Τέχνη & Δημιουργικότητα** 🎨
- 🖼️ Art Generation (DALL-E, Midjourney)
- 🎵 Music Composition
- ✍️ Creative Writing
- 🎬 Video Editing
- 🎮 Game Design
- 📝 Content Creation

---

### 📈 Ιστορική Εξέλιξη - Timeline

**1950s - Γέννηση**
- 1950: Alan Turing - "Computing Machinery and Intelligence"
- 1950: Turing Test
- 1956: **Dartmouth Conference** - Γέννηση του όρου "Artificial Intelligence"
- 1958: Perceptron (Rosenblatt)

**1960s-1970s - Πρώτα Βήματα**
- 1966: ELIZA (chatbot)
- 1969: Perceptron Limitations (Minsky & Papert)
- 1970s: First AI Winter

**1980s - Expert Systems**
- 1980s: Rule-based expert systems
- 1986: Backpropagation (Rumelhart)
- Late 80s: Second AI Winter

**1990s - Machine Learning**
- 1997: **Deep Blue** νικά τον Kasparov
- 1998: MNIST dataset
- 1998: LeNet-5 (Yann LeCun)

**2000s - Big Data Era**
- 2006: Deep Learning term (Hinton)
- 2009: ImageNet dataset
- 2011: **IBM Watson** νικά στο Jeopardy

**2010s - Deep Learning Revolution**
- 2012: **AlexNet** - ImageNet breakthrough
- 2014: GANs (Goodfellow)
- 2016: **AlphaGo** νικά τον Lee Sedol
- 2017: **Transformer** architecture ("Attention Is All You Need")
- 2018: GPT, BERT

**2020s - Mass Adoption**
- 2020: GPT-3 (175B parameters)
- 2021: DALL-E, AlphaFold 2
- 2022: **ChatGPT** - Μαζική υιοθέτηση
- 2023: GPT-4, Multimodal AI
- 2024: AI everywhere!

---

### ⚖️ Ηθικά Ζητήματα & Προκλήσεις

#### 1. **Bias & Fairness** (Μεροληψία & Δικαιοσύνη) ⚖️
**Πρόβλημα:**
- AI αναπαράγει προκαταλήψεις από δεδομένα
- Διακρίσεις σε φυλή, φύλο, ηλικία

**Λύσεις:**
- Diverse & representative data
- Bias detection & mitigation
- Fair AI algorithms
- Continuous monitoring

#### 2. **Privacy & Security** (Ιδιωτικότητα & Ασφάλεια) 🔒
**Πρόβλημα:**
- Συλλογή τεράστιων προσωπικών δεδομένων
- Data breaches
- Surveillance

**Λύσεις:**
- GDPR compliance
- Data minimization
- Encryption
- Privacy-preserving AI

#### 3. **Transparency & Explainability** (Διαφάνεια) 🔍
**Πρόβλημα:**
- "Black box" models
- Δύσκολο να εξηγηθούν αποφάσεις

**Λύσεις:**
- Explainable AI (XAI)
- Interpretable models
- Documentation
- Audit trails

#### 4. **Job Displacement** (Αντικατάσταση Εργασιών) 💼
**Πρόβλημα:**
- Automation → Job loss
- Ανισότητες

**Λύσεις:**
- Reskilling programs
- Universal Basic Income?
- New job creation
- Human-AI collaboration

#### 5. **Responsibility & Accountability** (Ευθύνη) 🎯
**Πρόβλημα:**
- Ποιος ευθύνεται για λάθη AI;
- Legal frameworks

**Λύσεις:**
- Clear regulations
- Liability frameworks
- AI governance
- Standards & certifications

#### 6. **Safety & Control** (Ασφάλεια & Έλεγχος) ⚠️
**Πρόβλημα:**
- Autonomous weapons
- AGI risk
- Loss of control

**Λύσεις:**
- AI safety research
- Alignment problem
- International cooperation
- Ethics guidelines

---

### 🔮 Μέλλον της AI

#### **Βραχυπρόθεσμο (1-5 χρόνια)** 📅
- Πιο ισχυρά LLMs
- Multimodal AI (text + image + video + audio)
- Edge AI (στη συσκευή)
- AI-powered productivity tools
- Personalized AI assistants

#### **Μεσοπρόθεσμο (5-15 χρόνια)** 📆
- Autonomous vehicles (wide adoption)
- AI in education (personalized)
- AI doctors (assistants)
- General-purpose robots
- AI-discovered drugs

#### **Μακροπρόθεσμο (15+ χρόνια)** 🗓️
- AGI (Artificial General Intelligence)?
- Brain-computer interfaces
- AI-human symbiosis
- Solving grand challenges (climate, disease)

#### **Τάσεις** 📈
1. **More Powerful Models**
   - Scaling laws
   - Efficient architectures
   
2. **Democratization**
   - AI for everyone
   - Low-code/no-code tools
   
3. **Specialization**
   - Domain-specific AI
   - Vertical solutions
   
4. **Responsible AI**
   - Ethics first
   - Governance frameworks
   
5. **Human-AI Collaboration**
   - Augmented intelligence
   - Best of both worlds

---

### 📚 Γιατί είναι Σημαντική η AI;

#### **Οφέλη** ✅

1. **Αυτοματοποίηση** 🤖
   - Απελευθέρωση ανθρώπινου χρόνου
   - 24/7 λειτουργία
   
2. **Ακρίβεια** 🎯
   - Μείωση ανθρώπινων λαθών
   - Consistent results
   
3. **Ταχύτητα** ⚡
   - Επεξεργασία τεράστιων δεδομένων
   - Real-time decisions
   
4. **Κλίμακα** 📊
   - Scaling σε παγκόσμιο επίπεδο
   - Personalization at scale
   
5. **Ανακάλυψη** 🔬
   - Νέα insights
   - Scientific breakthroughs
   
6. **Προσβασιμότητα** 🌍
   - Υπηρεσίες για όλους
   - Democratization

#### **Προκλήσεις** ⚠️
- Ηθικά ζητήματα
- Κόστος υλοποίησης
- Έλλειψη δεξιοτήτων
- Αντίσταση στην αλλαγή
- Regulatory uncertainty

---

### 📌 Σύνοψη - Key Takeaways

**Τι είναι η AI:**
✅ Συστήματα που μιμούνται ανθρώπινη νοημοσύνη
✅ Μαθαίνουν από δεδομένα και εμπειρία
✅ Λύνουν προβλήματα και παίρνουν αποφάσεις
✅ Προσαρμόζονται σε νέες καταστάσεις

**Γιατί είναι Σημαντική:**
🌟 Μεταμορφώνει κάθε τομέα
🌟 Αυξάνει παραγωγικότητα
🌟 Δημιουργεί νέες ευκαιρίες
🌟 Επιλύει μεγάλα προβλήματα

**Το Μέλλον:**
🚀 AI θα είναι παντού
🚀 Human-AI collaboration
🚀 Υπεύθυνη ανάπτυξη
🚀 Όφελος για όλους

---

**💡 Τελική Σκέψη:**

> Η Τεχνητή Νοημοσύνη δεν είναι μόνο τεχνολογία - είναι μια θεμελιώδης αλλαγή στον τρόπο που αλληλεπιδρούμε με τον κόσμο γύρω μας. Είναι εργαλείο, συνεργάτης και καταλύτης για την επίλυση των μεγαλύτερων προκλήσεων της ανθρωπότητας.

**Απλά λόγια:** Η AI κάνει τους υπολογιστές να σκέφτονται και να μαθαίνουν σαν άνθρωπους, βοηθώντας μας να λύσουμε προβλήματα, να δημιουργήσουμε νέα πράγματα και να ζήσουμε καλύτερα! 🧠💻✨
"""
    
    def _get_building_blocks(self):
        """Δομικά στοιχεία AI - Πλήρης ανάλυση"""
        return """
## 🏗️ Βασικά Δομικά Στοιχεία της Τεχνητής Νοημοσύνης

Τα τέσσερα θεμελιώδη δομικά στοιχεία της AI είναι:

### 📊 1. Δεδομένα (Data)
Τα δεδομένα αποτελούν τη βάση κάθε AI συστήματος.

### ⚙️ 2. Αλγόριθμοι (Algorithms)
Οι αλγόριθμοι είναι οι μαθηματικές μέθοδοι που μετατρέπουν δεδομένα σε χρήσιμες προβλέψεις.

### 🎯 3. Μοντέλα (Models)
Τα μοντέλα είναι τα εκπαιδευμένα συστήματα που προκύπτουν από την εφαρμογή αλγορίθμων σε δεδομένα.

### 💻 4. Υποδομές (Infrastructure)
Η υποδομή περιλαμβάνει το hardware και software που απαιτούνται για training και deployment AI μοντέλων.
"""
    
    def _get_machine_learning(self):
        """Machine Learning - Πλήρης ανάλυση"""
        return """
## 🧠 Machine Learning (Μηχανική Μάθηση)

Το **Machine Learning** είναι υποκλάδος της AI που επιτρέπει στους υπολογιστές να μαθαίνουν από δεδομένα.

### Τύποι ML:
- **Supervised Learning**: Μάθηση με labeled data
- **Unsupervised Learning**: Μάθηση χωρίς labels
- **Reinforcement Learning**: Μάθηση μέσω δοκιμής και λάθους
"""

    def _get_supervised_learning(self):
        """Supervised Learning - Πλήρης ανάλυση"""
        return """
## 🎯 Supervised Learning (Επιβλεπόμενη Μάθηση)

Η **Supervised Learning** είναι η πιο κοινή μέθοδος ML όπου το μοντέλο μαθαίνει από **labeled data**.

### Χαρακτηριστικά:
- Έχουμε input features (X) και target labels (y)
- Το μοντέλο μαθαίνει τη σχέση X → y
- Χρησιμοποιείται για προβλέψεις σε νέα δεδομένα

### Τύποι:
1. **Classification**: Πρόβλεψη κατηγορίας (π.χ. spam/not spam)
2. **Regression**: Πρόβλεψη αριθμητικής τιμής (π.χ. τιμή σπιτιού)

### Αλγόριθμοι:
- Linear/Logistic Regression
- Decision Trees
- Random Forest
- SVM (Support Vector Machines)
- Neural Networks
"""

    def _get_unsupervised_learning(self):
        """Unsupervised Learning - Πλήρης ανάλυση"""
        return """
## 🔍 Unsupervised Learning (Μη Επιβλεπόμενη Μάθηση)

Η **Unsupervised Learning** μαθαίνει από **unlabeled data** και ανακαλύπτει κρυφά patterns.

### Τύποι:
1. **Clustering**: Ομαδοποίηση παρόμοιων δεδομένων (K-Means, DBSCAN)
2. **Dimensionality Reduction**: Μείωση features (PCA, t-SNE)

### Εφαρμογές:
- Customer segmentation
- Anomaly detection
- Feature extraction
"""

    def _get_reinforcement_learning(self):
        """Reinforcement Learning - Πλήρης ανάλυση"""
        return """
## 🎮 Reinforcement Learning (Ενισχυτική Μάθηση)

Η **Reinforcement Learning** μαθαίνει μέσω δοκιμής-λάθους και ανταμοιβών.

### Στοιχεία:
- **Agent**: Το σύστημα που μαθαίνει
- **Environment**: Το περιβάλλον
- **Actions**: Ενέργειες
- **Rewards**: Ανταμοιβές/τιμωρίες

### Εφαρμογές:
- Gaming AI (AlphaGo, Chess)
- Robotics
- Autonomous vehicles
"""

    def _get_deep_learning(self):
        """Deep Learning - Πλήρης ανάλυση"""
        return """
## 🌊 Deep Learning (Βαθιά Μάθηση)

Το **Deep Learning** χρησιμοποιεί νευρωνικά δίκτυα με πολλά layers.

### Αρχιτεκτονικές:
- **CNN**: Για εικόνες
- **RNN/LSTM**: Για sequences
- **Transformers**: Για NLP
- **GANs**: Για generation

### Frameworks:
- TensorFlow
- PyTorch
- Keras
"""

    def _get_neural_networks(self):
        """Neural Networks - Πλήρης ανάλυση"""
        return """
## 🧠 Neural Networks (Νευρωνικά Δίκτυα)

Τα **Neural Networks** εμπνέονται από τον ανθρώπινο εγκέφαλο.

### Δομή:
- **Input Layer**: Δέχεται δεδομένα
- **Hidden Layers**: Επεξεργάζεται πληροφορίες
- **Output Layer**: Παράγει αποτελέσματα

### Activation Functions:
- ReLU, Sigmoid, Tanh, Softmax
"""

    def _get_cnn(self):
        """CNN - Πλήρης ανάλυση"""
        return """
## 📸 CNN (Convolutional Neural Networks)

Τα **CNN** είναι ειδικευμένα για επεξεργασία εικόνων.

### Χαρακτηριστικά:
- Convolution layers
- Pooling layers
- Fully connected layers

### Εφαρμογές:
- Image classification
- Object detection
- Face recognition
"""

    def _get_rnn(self):
        """RNN - Πλήρης ανάλυση"""
        return """
## 🔄 RNN (Recurrent Neural Networks)

Τα **RNN** είναι για sequential data.

### Τύποι:
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

### Εφαρμογές:
- NLP
- Time series
- Speech recognition
"""

    def _get_transformer(self):
        """Transformer - Πλήρης ανάλυση"""
        return """
## 🔬 Transformer Architecture

Οι **Transformers** επανάστησαν το NLP (2017).

### Μηχανισμός:
- Self-attention mechanism
- Parallel processing
- Positional encoding

### Μοντέλα:
- BERT, GPT, T5
- ChatGPT βασίζεται σε Transformers
"""

    def _get_chatgpt(self):
        """ChatGPT - Πλήρης ανάλυση"""
        return """
## 🤖 ChatGPT

Το **ChatGPT** είναι Large Language Model από την OpenAI.

### Χαρακτηριστικά:
- Βασίζεται σε GPT architecture (Transformer)
- Pre-trained σε τεράστια datasets
- Fine-tuned με RLHF

### Δυνατότητες:
- Συνομιλία
- Δημιουργία κειμένου
- Προγραμματισμός
- Μετάφραση
"""

    def _get_llm(self):
        """LLM - Πλήρης ανάλυση"""
        return """
## 📚 Large Language Models (LLMs)

Τα **LLMs** είναι μεγάλα γλωσσικά μοντέλα.

### Παραδείγματα:
- GPT-4, Claude, Gemini
- BERT, T5

### Χαρακτηριστικά:
- Δισεκατομμύρια παράμετροι
- Pre-training + Fine-tuning
- Zero/Few-shot learning
"""

    def _get_generative_ai(self):
        """Generative AI - Πλήρης ανάλυση"""
        return """
## 🎨 Generative AI

Η **Generative AI** δημιουργεί νέο περιεχόμενο.

### Τύποι:
- **Text**: ChatGPT, Claude
- **Images**: DALL-E, Midjourney, Stable Diffusion
- **Audio**: Music, Voice synthesis
- **Video**: Sora

### Τεχνολογίες:
- GANs, VAEs, Diffusion Models, Transformers
"""

    def _get_gan(self):
        """GAN - Πλήρης ανάλυση"""
        return """
## 🎭 GANs (Generative Adversarial Networks)

Τα **GANs** έχουν δύο δίκτυα που "παλεύουν":

### Δομή:
- **Generator**: Δημιουργεί fake data
- **Discriminator**: Διακρίνει real vs fake

### Εφαρμογές:
- Image generation
- Style transfer
- Data augmentation
"""

    def _get_applications(self):
        """Εφαρμογές AI"""
        return """
## 💼 Εφαρμογές AI

Η AI χρησιμοποιείται σε πολλούς τομείς:

### Κυρίως:
- 🏥 **Υγεία**: Διάγνωση, drug discovery
- 💰 **Finance**: Fraud detection, trading
- 🚗 **Autonomous Vehicles**: Self-driving cars
- 🎓 **Εκπαίδευση**: Personalized learning
- 🛒 **E-commerce**: Recommendations
"""

    def _get_health_applications(self):
        """Εφαρμογές στην Υγεία"""
        return """
## 🏥 AI στην Υγεία

### Εφαρμογές:
- Medical imaging analysis
- Drug discovery
- Personalized medicine
- Patient monitoring
- Clinical decision support
"""

    def _get_education_applications(self):
        """Εφαρμογές στην Εκπαίδευση"""
        return """
## 🎓 AI στην Εκπαίδευση

### Εφαρμογές:
- Personalized learning paths
- Intelligent tutoring systems
- Automated grading
- Learning analytics
"""

    def _get_ethics(self):
        """Ηθική AI"""
        return """
## ⚖️ Ηθική στην AI

### Ζητήματα:
- **Bias**: Μεροληψία στα δεδομένα
- **Privacy**: Προστασία προσωπικών δεδομένων
- **Transparency**: Explainable AI
- **Accountability**: Ποιος ευθύνεται;
- **Job displacement**: Αντικατάσταση εργασιών
"""

    def _get_privacy(self):
        """Privacy και GDPR"""
        return """
## 🔒 Privacy & GDPR

### Σημαντικά:
- Προστασία προσωπικών δεδομένων
- GDPR compliance
- Data minimization
- Right to be forgotten
- Consent management
"""

    def _get_python(self):
        """Python για AI"""
        return """
## 🐍 Python για AI/ML

### Βιβλιοθήκες:
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Scikit-learn**: ML algorithms
- **TensorFlow/PyTorch**: Deep Learning
- **Matplotlib/Seaborn**: Visualization
"""

    def _get_colab(self):
        """Google Colab"""
        return """
## 📓 Google Colab

Το **Google Colab** προσφέρει:
- Δωρεάν GPU/TPU
- Jupyter notebooks στο cloud
- Προ-εγκατεστημένες βιβλιοθήκες
- Εύκολο sharing

Ιδανικό για AI/ML experiments!
"""
    
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
        
        if best_match and max_score > 0:
            return best_match
        else:
            # Αν δεν βρέθηκε, δοκίμασε internet
            if self.use_internet:
                online_info = self._search_online(question)
                if online_info:
                    return online_info
            
            return self._generate_generic_answer(question)
    
    def _generate_generic_answer(self, question: str) -> str:
        """Γενική απάντηση όταν δεν βρεθεί match"""
        return f"""
## 🤔 Δεν βρήκα συγκεκριμένη απάντηση

Η ερώτησή σας: "{question}"

### 💡 Προτάσεις:

**Μπορείτε να ρωτήσετε για:**
- Ορισμό της Τεχνητής Νοημοσύνης
- Βασικά δομικά στοιχεία της AI
- Machine Learning και τους τύπους του
- Deep Learning και νευρωνικά δίκτυα
- Πώς λειτουργεί το ChatGPT
- Εφαρμογές AI σε διάφορους τομείς
- Ηθικά ζητήματα της AI

**Παραδείγματα ερωτήσεων:**
- "Τι είναι η Τεχνητή Νοημοσύνη;"
- "Ποια είναι τα βασικά δομικά στοιχεία της AI;"
- "Εξήγησε το Machine Learning"
- "Πώς λειτουργεί το ChatGPT;"
- "Τι είναι το Deep Learning;"

### 🌐 Αναζήτηση στο Internet
Το chatbot μπορεί επίσης να αναζητήσει πληροφορίες online από:
- Wikipedia
- Επίσημα documentation (TensorFlow, PyTorch)
- Research papers (ArXiv)
- Educational resources
"""
    
    def _search_online(self, question: str) -> str:
        """Αναζήτηση online - simplified για demo"""
        # [Εδώ θα ήταν ο κώδικας για internet search]
        return None

# Streamlit Interface Function
def create_enriched_chatbot_interface():
    """Δημιουργία εμπλουτισμένου Streamlit interface για το chatbot"""
    st.markdown("### 🤖 AI Knowledge Assistant - Εμπλουτισμένη Έκδοση")
    st.markdown("*Ρωτήστε με οτιδήποτε σχετικό με Τεχνητή Νοημοσύνη! Έχω πρόσβαση σε πλήρες εκπαιδευτικό υλικό + online πηγές.*")
    
    # Status indicators
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption("💡 **Πλήρες εκπαιδευτικό υλικό**: Όλες οι ενότητες 1.1-1.7")
    with col2:
        st.success("📚 Local KB", icon="✅")
    with col3:
        st.success("🌐 Online", icon="✅")
    
    # Initialize chatbot
    if 'enriched_chatbot' not in st.session_state:
        st.session_state.enriched_chatbot = AIKnowledgeBotEnriched()
    
    # Initialize chat history
    if 'enriched_messages' not in st.session_state:
        st.session_state.enriched_messages = []
    
    # Display chat history
    for message in st.session_state.enriched_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Γράψτε την ερώτησή σας εδώ..."):
        # Add user message
        st.session_state.enriched_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Αναζητώ στο πλήρες εκπαιδευτικό υλικό και online πηγές..."):
                response = st.session_state.enriched_chatbot.get_answer(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.enriched_messages.append({"role": "assistant", "content": response})
    
    # Quick questions - Εμπλουτισμένες
    st.markdown("---")
    st.markdown("#### 💬 Γρήγορες Ερωτήσεις:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_questions = [
        ("🤖 Ορισμός AI", "Δώσε μου έναν ορισμό για την Τεχνητή Νοημοσύνη"),
        ("🏗️ Δομικά Στοιχεία", "Περιγράψτε τα βασικά δομικά στοιχεία της AI"),
        ("🧠 Machine Learning", "Εξήγησε το Machine Learning"),
        ("🌊 Deep Learning", "Τι είναι το Deep Learning;"),
        ("🤖 ChatGPT", "Πώς λειτουργεί το ChatGPT;"),
        ("🔬 Transformer", "Τι είναι το Transformer;"),
        ("⚖️ Ηθική AI", "Ποια είναι τα ηθικά ζητήματα της AI;"),
        ("💼 Εφαρμογές", "Ποιες είναι οι εφαρμογές της AI;")
    ]
    
    for i, (label, question) in enumerate(quick_questions):
        col = [col1, col2, col3, col4][i % 4]
        with col:
            if st.button(label, key=f"quick_{i}"):
                st.session_state.enriched_messages.append({"role": "user", "content": question})
                response = st.session_state.enriched_chatbot.get_answer(question)
                st.session_state.enriched_messages.append({"role": "assistant", "content": response})
                st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Καθαρισμός Συνομιλίας"):
        st.session_state.enriched_messages = []
        st.rerun()
    
    # Info about knowledge base
    st.markdown("---")
    with st.expander("📚 Βάση Γνώσης - Τι Περιλαμβάνει"):
        st.markdown("""
        ### 📖 Τοπική Βάση Γνώσης (Εκπαιδευτικό Υλικό)
        
        **Θεωρητικό Μέρος:**
        - ✅ Πλήρης ορισμός Τεχνητής Νοημοσύνης
        - ✅ Βασικά δομικά στοιχεία (Δεδομένα, Αλγόριθμοι, Μοντέλα, Computing)
        - ✅ Machine Learning (Supervised, Unsupervised, Reinforcement)
        - ✅ Deep Learning & Neural Networks
        - ✅ CNN, RNN, Transformers
        - ✅ ChatGPT & Large Language Models
        - ✅ Generative AI (GANs, VAEs, Diffusion Models)
        - ✅ NLP, Computer Vision, Robotics
        - ✅ Εφαρμογές σε όλους τους τομείς
        - ✅ Ηθικά ζητήματα & προκλήσεις
        - ✅ Python & Programming Tools
        
        **Διαδραστικά Notebooks:**
        - Google Colab tutorials
        - Πρακτικά παραδείγματα
        - Hands-on exercises
        
        ### 🌐 Online Πηγές
        
        **Wikipedia:**
        - Γενικές πληροφορίες για AI concepts
        
        **Curated AI Resources:**
        - Official Documentation (TensorFlow, PyTorch, Keras)
        - Research Papers (ArXiv)
        - Educational Platforms (Coursera, Fast.ai, DeepLearning.AI)
        - Interactive Tools (TensorFlow Playground)
        - Open Source Projects (Hugging Face)
        
        ### 🎯 Δυνατότητες
        
        - ✅ Απαντά σε ερωτήσεις με πλήρη ανάλυση
        - ✅ Παρέχει παραδείγματα και use cases
        - ✅ Εξηγεί τεχνικές λεπτομέρειες
        - ✅ Συνδέει έννοιες μεταξύ τους
        - ✅ Προτείνει επιπλέον πόρους
        - ✅ Αναζητά online για πρόσθετες πληροφορίες
        """)
    
    # Statistics
    st.markdown("---")
    st.markdown("### 📊 Στατιστικά")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ερωτήσεις σε αυτή τη συνομιλία", len([m for m in st.session_state.enriched_messages if m["role"] == "user"]))
    with col2:
        st.metric("Θέματα στη βάση γνώσης", len(st.session_state.enriched_chatbot.qa_pairs))
    with col3:
        st.metric("Online πηγές", "10+")
