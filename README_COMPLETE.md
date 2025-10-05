# 🤖 AI Training App - Πλήρης Εκπαιδευτική Εφαρμογή

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 📋 Περιγραφή

Πλήρης διαδραστική εφαρμογή εκπαίδευσης στην Τεχνητή Νοημοσύνη με:
- 📚 Θεωρητικό περιεχόμενο (Ενότητες 1.1-1.7)
- 🐍 Πρακτικά παραδείγματα Python
- 🔬 Διαδραστικές εξομοιώσεις ML
- ✅ Κουίζ αυτοαξιολόγησης
- 💡 Διαδραστικές ασκήσεις
- 🤖 **AI Chatbot** με πλήρη γνώση AI concepts
- 📖 Πόροι και Google Colab notebooks

## 🚀 Γρήγορη Εκκίνηση

### Online (Recommended)

Η εφαρμογή τρέχει online στο Streamlit Cloud:

👉 **[Ανοίξτε την Εφαρμογή](https://your-app-url.streamlit.app)**

### Local Installation

```bash
# Clone το repository
git clone https://github.com/krimits/ai_training_app.git
cd ai_training_app

# Εγκατάσταση dependencies
pip install -r requirements.txt

# Εκτέλεση εφαρμογής
streamlit run ai_training_app.py
```

Η εφαρμογή θα ανοίξει στο browser σας στο `http://localhost:8501`

## 📦 Dependencies

```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
seaborn>=0.12.0
```

## 🎯 Χαρακτηριστικά

### 📚 Tab 1: Περιεχόμενο
- Πλήρης θεωρία για AI, ML, Deep Learning
- Expandable sections με εις βάθος εξηγήσεις
- Concept explainers για κάθε όρο
- Hyperlinks σε πρόσθετους πόρους

### 🐍 Tab 2: Παραδείγματα Python
- **Logistic Regression** για binary classification
- **K-Means Clustering** για unsupervised learning
- **Neural Networks** με scikit-learn

### 🔬 Tab 3: Εξομοιώσεις AI
- Επίδραση θορύβου στην ακρίβεια
- Overfitting vs Underfitting visualization
- Επίδραση μεγέθους dataset
- Decision boundary visualization

### ✅ Tab 4: Κουίζ
- 15+ ερωτήσεις σε 5 κατηγορίες
- Άμεση ανατροφοδότηση
- Εξηγήσεις για κάθε απάντηση

### 💡 Tab 5: Διαδραστικές Ασκήσεις
- Google Colab notebooks (Beginner & Advanced)
- In-app exercises:
  - Πρόβλεψη τιμών (Regression)
  - Image Classification Simulation
  - Sentiment Analysis
  - Recommendation System

### 🤖 Tab 6: AI Chatbot
**✨ ΝΕΟ! Εμπλουτισμένο Chatbot**

Intelligent assistant που απαντά σε ερωτήσεις για:
- ✅ Ορισμοί και βασικές έννοιες AI
- ✅ Machine Learning (Supervised, Unsupervised, Reinforcement)
- ✅ Deep Learning & Neural Networks
- ✅ ChatGPT & Large Language Models
- ✅ Εφαρμογές σε όλους τους τομείς
- ✅ Ηθικά ζητήματα & Privacy

**Δυνατότητες:**
- 🔍 Έξυπνη αναζήτηση με keywords
- 💬 Φυσική γλώσσα (Ελληνικά & Αγγλικά)
- 📊 Πλούσιες, δομημένες απαντήσεις
- 🎯 Quick questions για γρήγορη πρόσβαση
- 📚 7 κύρια θέματα coverage

**Παραδείγματα Ερωτήσεων:**
- "Τι είναι η Τεχνητή Νοημοσύνη;"
- "Εξήγησε το Machine Learning"
- "Πώς λειτουργεί το ChatGPT;"
- "Ποια είναι τα ηθικά ζητήματα της AI;"

### 📖 Tab 7: Πόροι
- Online courses (Coursera, Fast.ai, DeepLearning.AI)
- Recommended books
- Frameworks & Tools (TensorFlow, PyTorch)
- Datasets (Kaggle, UCI ML Repository)
- Communities (Kaggle, GitHub, Reddit)

## 🏗️ Δομή Project

```
ai_training_app/
├── ai_training_app.py          # Main Streamlit app
├── chatbot_simple.py            # AI Chatbot module
├── requirements.txt             # Python dependencies
├── pdf_content.txt              # Educational content (extracted from PDF)
├── sample_data.csv              # Sample dataset
├── README.md                    # This file
├── CHATBOT_DOCS.md              # Chatbot documentation
├── COLAB_NOTEBOOKS.md           # Google Colab notebooks info
└── Εφαρμογές τεχνητής νοημοσύνης και ChatGPT σε κρίσιμους τομείς.pdf
```

## 🤖 Chatbot Technical Details

### Αρχιτεκτονική
- **Keyword-based matching** με scoring system
- **7 κύρια θέματα** στη βάση γνώσης
- **Markdown formatting** για πλούσιες απαντήσεις
- **Context-aware** responses

### Supported Topics
1. AI Definition & Concepts
2. Building Blocks of AI
3. Machine Learning (Supervised, Unsupervised, Reinforcement)
4. Deep Learning & Neural Networks
5. ChatGPT & LLMs
6. AI Applications across domains
7. Ethics & Responsible AI

### Code Example
```python
from chatbot_simple import AIKnowledgeBot

# Initialize bot
bot = AIKnowledgeBot()

# Ask a question
answer = bot.get_answer("Τι είναι η Τεχνητή Νοημοσύνη;")
print(answer)
```

## 📊 Features Comparison

| Feature | Basic | Current Version |
|---------|-------|-----------------|
| Theoretical Content | ✅ | ✅ Enhanced with expandables |
| Python Examples | ✅ | ✅ |
| ML Simulations | ✅ | ✅ |
| Quiz | ✅ | ✅ |
| Interactive Exercises | ✅ | ✅ With Colab links |
| **AI Chatbot** | ❌ | ✅ **NEW!** |
| Resources | ✅ | ✅ |

## 🎓 Educational Content Coverage

### Ενότητες (Sections)
- **1.1** Εισαγωγή - Τι είναι η Τεχνητή Νοημοσύνη
- **1.2** Κύρια Δομικά Στοιχεία της ΤΝ
- **1.3** Βασικά Ιστορικά Επιτεύγματα
- **1.4** Τεχνητή Νοημοσύνη: Εφαρμογές και Εξελίξεις
- **1.5** Βασικές Έννοιες - Πλαίσιο - Κανόνες
- **1.6** Πώς Λειτουργεί το ChatGPT
- **1.7** Βασικές Αρχές AI και Εφαρμογές

### Τεχνολογίες
- Machine Learning (ML)
- Deep Learning (DL)
- Natural Language Processing (NLP)
- Computer Vision (CV)
- Robotics
- Expert Systems
- Generative AI

## 🛠️ Development

### Running Locally

```bash
# Development mode με auto-reload
streamlit run ai_training_app.py --server.runOnSave true
```

### Adding New Chatbot Topics

Edit `chatbot_simple.py`:

```python
def _create_qa_database(self):
    return {
        "new_topic": {
            "keywords": ["keyword1", "keyword2"],
            "answer": """
            ## Your Answer Here
            
            Markdown formatted content...
            """
        }
    }
```

## 🚀 Deployment στο Streamlit Cloud

### Βήμα 1: Prepare Repository

Βεβαιωθείτε ότι έχετε:
- ✅ `requirements.txt` με όλα τα dependencies
- ✅ `ai_training_app.py` (main file)
- ✅ `chatbot_simple.py`
- ✅ Όλα τα αρχεία στο GitHub repository

### Βήμα 2: Deploy

1. Πηγαίνετε στο [share.streamlit.io](https://share.streamlit.io)
2. Συνδεθείτε με GitHub account
3. Κλικ στο **"New app"**
4. Επιλέξτε:
   - Repository: `krimits/ai_training_app`
   - Branch: `main`
   - Main file: `ai_training_app.py`
5. Κλικ **"Deploy"**

### Βήμα 3: Τελειώσατε! 🎉

Η εφαρμογή θα είναι διαθέσιμη σε URL όπως:
`https://krimits-ai-training-app-main.streamlit.app`

## 📝 Usage Examples

### Θεωρία
1. Πηγαίνετε στο tab "📚 Περιεχόμενο"
2. Ανοίξτε expandable sections για εις βάθος εξήγηση
3. Κλικ σε concept explainers για επιπλέον λεπτομέρειες

### Python Παραδείγματα
1. Tab "🐍 Παραδείγματα Python"
2. Προσαρμόστε parameters με sliders
3. Κλικ "Εκπαίδευση Μοντέλου"
4. Δείτε visualizations και metrics

### Chatbot
1. Tab "🤖 AI Chatbot"
2. Γράψτε ερώτηση στο chat input
3. ή Κλικ σε Quick Question button
4. Διαβάστε την πλήρη, δομημένη απάντηση
5. Ρωτήστε follow-up ερωτήσεις!

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork το repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push στο branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes.

## 👤 Author

**Your Name**
- GitHub: [@krimits](https://github.com/krimits)
- Repository: [ai_training_app](https://github.com/krimits/ai_training_app)

## 🙏 Acknowledgments

- Βασισμένο στο εκπαιδευτικό υλικό: "Εφαρμογές Τεχνητής Νοημοσύνης και ChatGPT σε Κρίσιμους Τομείς"
- Streamlit για το amazing framework
- scikit-learn, matplotlib, pandas για τα ML tools
- Η AI community για inspiration

## 📞 Support & Contact

Για ερωτήσεις, issues, ή suggestions:
- 🐛 [GitHub Issues](https://github.com/krimits/ai_training_app/issues)
- 📧 Email: your-email@example.com
- 💬 Discussions: [GitHub Discussions](https://github.com/krimits/ai_training_app/discussions)

## 🎯 Roadmap

### Completed ✅
- [x] Βασική εφαρμογή με 7 tabs
- [x] Python examples & ML simulations
- [x] Quiz system
- [x] Google Colab integration
- [x] AI Chatbot με πλήρη γνώση

### Planned 🚧
- [ ] More chatbot topics (Python, frameworks)
- [ ] Video tutorials integration
- [ ] User progress tracking
- [ ] Certificate generation
- [ ] Multilingual support (English)
- [ ] Mobile-optimized UI
- [ ] API για chatbot

## 📈 Stats

- **Lines of Code**: ~3000+
- **Chatbot Topics**: 7
- **Interactive Examples**: 10+
- **Quiz Questions**: 15+
- **Educational Sections**: 7 major sections
- **Google Colab Notebooks**: 6 linked

---

## 🎉 Getting Started - Quick Guide

### 1️⃣ Visit the App
👉 https://your-app-url.streamlit.app

### 2️⃣ Start with Περιεχόμενο (Content)
Learn the theory behind AI concepts

### 3️⃣ Try Python Examples
See ML in action with live code

### 4️⃣ Test Knowledge with Quiz
Check your understanding

### 5️⃣ Ask the Chatbot!
Get answers to any AI question

### 6️⃣ Do Colab Exercises
Hands-on practice with real code

---

**Made with ❤️ and Python**

**Powered by:**
- 🎈 Streamlit
- 🐍 Python
- 🧠 AI/ML libraries
- ☕ Coffee

---

*Last Updated: January 2025*
