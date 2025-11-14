# 🏗️ Αρχιτεκτονική AI Training App - UML Διαγράμματα

## 📋 Περιεχόμενα

1. [Εισαγωγή](#εισαγωγή)
2. [Class Diagram - Chatbot Module](#class-diagram---chatbot-module)
3. [Sequence Diagram - Chatbot Interaction](#sequence-diagram---chatbot-interaction)
4. [Component Diagram - System Architecture](#component-diagram---system-architecture)
5. [Activity Diagram - Main Application Flow](#activity-diagram---main-application-flow)
6. [Use Case Diagram - User Interactions](#use-case-diagram---user-interactions)
7. [Deployment Diagram - Deployment Architecture](#deployment-diagram---deployment-architecture)
8. [Τεχνική Υλοποίηση](#τεχνική-υλοποίηση)

---

## Εισαγωγή

Αυτό το έγγραφο παρουσιάζει την **αρχιτεκτονική του AI Training App** μέσω **UML διαγραμμάτων**. Κάθε διάγραμμα προσφέρει διαφορετική οπτική του συστήματος:

- **Class Diagram**: Δομή κλάσεων και σχέσεις
- **Sequence Diagram**: Ροή αλληλεπιδράσεων
- **Component Diagram**: Αρχιτεκτονική συστατικών
- **Activity Diagram**: Ροή λειτουργιών
- **Use Case Diagram**: Λειτουργικές απαιτήσεις
- **Deployment Diagram**: Αρχιτεκτονική ανάπτυξης

### 📁 Αρχεία Διαγραμμάτων

Όλα τα διαγράμματα είναι σε **PlantUML format** και βρίσκονται στο φάκελο `/diagrams/`:

```
diagrams/
├── class_diagram_chatbot.puml
├── sequence_diagram_chatbot.puml
├── component_diagram_architecture.puml
├── activity_diagram_main_flow.puml
├── use_case_diagram.puml
└── deployment_diagram.puml
```

### 🔧 Προβολή Διαγραμμάτων

Για να δείτε τα διαγράμματα:

1. **Online**: Χρησιμοποιήστε το [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
2. **VS Code**: Εγκαταστήστε το extension "PlantUML"
3. **IntelliJ IDEA**: Ενσωματωμένη υποστήριξη
4. **GitHub**: Τα .puml αρχεία μπορούν να προβληθούν με extensions

---

## Class Diagram - Chatbot Module

### 📖 Περιγραφή

Το **Class Diagram** δείχνει τη δομή της κλάσης `AIKnowledgeBot` που αποτελεί τον πυρήνα του AI chatbot.

### 🎯 Βασικά Στοιχεία

#### Κλάση: AIKnowledgeBot

**Attributes (Ιδιότητες):**
- `knowledge_base`: Περιεχόμενο από το PDF (731KB)
- `qa_pairs`: Λεξικό με προ-ορισμένες ερωτήσεις-απαντήσεις
- `use_internet`: Flag για internet access
- `wikipedia_api`: URL του Wikipedia API
- `sources_used`: Λίστα πηγών που χρησιμοποιήθηκαν

**Public Methods:**
- `get_answer(question: str)`: Κύρια μέθοδος για λήψη απάντησης

**Private Methods:**
- `_load_knowledge()`: Φόρτωση PDF content
- `_create_qa_database()`: Δημιουργία QA database
- `_generate_generic_answer()`: Γενική απάντηση από KB
- `_search_online()`: Web search
- `_search_wikipedia()`: Wikipedia search
- `_search_curated_sources()`: Αναζήτηση σε επιλεγμένες πηγές
- `_extract_topics()`: Εξαγωγή θεμάτων από ερώτηση

#### External Dependencies

- **KnowledgeBase**: Το αρχείο `pdf_content.txt`
- **WikipediaAPI**: External Wikipedia REST API
- **WebSearch**: Web search functionality

### 📝 PlantUML Source

```plantuml
diagrams/class_diagram_chatbot.puml
```

### 💡 Σημαντικά Σημεία

1. **Encapsulation**: Οι περισσότερες μέθοδοι είναι private (prefix `_`)
2. **Single Responsibility**: Η κλάση έχει σαφή ευθύνη - να απαντά σε ερωτήσεις
3. **Dependency Injection**: Το knowledge file path περνάει στον constructor
4. **Multiple Sources**: Υποστήριξη πολλαπλών πηγών (PDF, Wikipedia, Web)

---

## Sequence Diagram - Chatbot Interaction

### 📖 Περιγραφή

Το **Sequence Diagram** δείχνει τη **χρονική ροή** της αλληλεπίδρασης μεταξύ χρήστη και chatbot.

### 🎯 Ροή Διαδικασίας

#### 1. Αρχικοποίηση (Initialization)
```
User → Streamlit UI → AIKnowledgeBot
    ↓
Knowledge Base loaded
    ↓
QA Database created
```

#### 2. Υποβολή Ερώτησης (Question Submission)
```
User → Streamlit UI → AIKnowledgeBot.get_answer()
    ↓
Extract topics from question
    ↓
Search in QA database
```

#### 3. Αναζήτηση Απάντησης (Answer Search)

**Cascade Search Strategy:**

1. **QA Database** - Προ-ορισμένες απαντήσεις
   - Άμεση επιστροφή αν βρεθεί

2. **Knowledge Base** - Αναζήτηση στο PDF content
   - Generate generic answer από context

3. **Wikipedia** - External API call
   - Αναζήτηση σχετικών άρθρων

4. **Web Search** - Last resort
   - Γενική αναζήτηση στο διαδίκτυο

#### 4. Επιστροφή Απόκρισης (Response Return)
```
AIKnowledgeBot → Streamlit UI → User
    ↓
Display answer + sources
```

### 📝 PlantUML Source

```plantuml
diagrams/sequence_diagram_chatbot.puml
```

### 💡 Σημαντικά Σημεία

1. **Fallback Mechanism**: Σταδιακή αναζήτηση από local σε remote πηγές
2. **Source Tracking**: Καταγραφή πηγών για διαφάνεια
3. **Error Handling**: Alternative flows σε κάθε βήμα
4. **Async Behavior**: External API calls

---

## Component Diagram - System Architecture

### 📖 Περιγραφή

Το **Component Diagram** δείχνει την **αρχιτεκτονική του συστήματος** σε επίπεδο components.

### 🎯 Layers (Επίπεδα)

#### 1. Frontend Layer
- **Streamlit UI**: User interface
- **Session State**: State management

#### 2. Application Layer
- **Main App** (`ai_training_app.py`): Core application
- **Content Sections**: Θεωρητικό υλικό
- **Examples Module**: Python examples
- **Quiz Module**: Κουίζ αυτοαξιολόγησης
- **Exercises Module**: Διαδραστικές ασκήσεις

#### 3. AI Chatbot Layer
- **AIKnowledgeBot**: Chatbot logic
- **QA Database**: Προ-ορισμένες Q&A

#### 4. Data Layer
- **Knowledge Base**: `pdf_content.txt` (731KB)
- **Sample Data**: `sample_data.csv`

#### 5. External Services
- **Wikipedia API**: Knowledge retrieval
- **Web Search**: Online search

#### 6. ML Libraries
- **scikit-learn**: ML algorithms
- **NumPy/Pandas**: Data processing
- **Matplotlib/Seaborn**: Visualization

### 📊 Component Relationships

```
Streamlit UI
    ↓
Main App
    ├─→ Content (Theory)
    ├─→ Examples (Python + ML Libraries)
    ├─→ Quiz
    ├─→ Exercises (Sample Data + ML)
    └─→ Chatbot (Knowledge Base + External APIs)
```

### 📝 PlantUML Source

```plantuml
diagrams/component_diagram_architecture.puml
```

### 💡 Σημαντικά Σημεία

1. **Layered Architecture**: Σαφής διαχωρισμός επιπέδων
2. **Separation of Concerns**: Κάθε component έχει συγκεκριμένο ρόλο
3. **External Dependencies**: Σαφείς εξαρτήσεις από external services
4. **Modular Design**: Κάθε module μπορεί να αναπτυχθεί ανεξάρτητα

---

## Activity Diagram - Main Application Flow

### 📖 Περιγραφή

Το **Activity Diagram** δείχνει τη **ροή δραστηριοτήτων** στην κύρια εφαρμογή.

### 🎯 Ροή Εφαρμογής

#### 1. Εκκίνηση
```
Start → Load Streamlit App → Display Main Menu
```

#### 2. Tab Selection

**📚 Περιεχόμενο Tab:**
- 7 Θεωρητικές ενότητες
- Expandable sections
- Concept explainers

**🐍 Python Examples Tab:**
- Logistic Regression
- K-Means Clustering
- Neural Networks
- Interactive parameters

**🔬 Εξομοιώσεις Tab:**
- Noise impact
- Overfitting/Underfitting
- Dataset size effects
- Decision boundaries

**✅ Κουίζ Tab:**
- 15 ερωτήσεις
- Immediate feedback
- Explanations

**💡 Ασκήσεις Tab:**
- House price prediction
- Image classification
- Sentiment analysis
- Recommendation system

**🤖 AI Chatbot Tab:**
- Question submission loop
- Multi-source search
- Answer display

**📖 Πόροι Tab:**
- Colab notebooks
- External links
- Documentation

#### 3. User Interaction Loop
```
Select Tab → Interact → Continue? → Yes → Back to Menu
                                  → No → Exit
```

### 📝 PlantUML Source

```plantuml
diagrams/activity_diagram_main_flow.puml
```

### 💡 Σημαντικά Σημεία

1. **Multi-Tab Design**: Οργάνωση περιεχομένου σε tabs
2. **Interactive Elements**: Sliders, buttons, inputs
3. **Feedback Loop**: Άμεση ανατροφοδότηση στον χρήστη
4. **Non-Linear Navigation**: Ελεύθερη μετακίνηση μεταξύ tabs

---

## Use Case Diagram - User Interactions

### 📖 Περιγραφή

Το **Use Case Diagram** δείχνει τις **λειτουργικές απαιτήσεις** και τους **actors** του συστήματος.

### 🎯 Actors (Δρώντα Πρόσωπα)

1. **Μαθητής/Φοιτητής**: Κύριος χρήστης που μαθαίνει AI
2. **Επαγγελματίας**: Upskilling/reskilling
3. **Εκπαιδευτικός**: Χρήση ως εκπαιδευτικό εργαλείο

### 🎯 Use Cases (Περιπτώσεις Χρήσης)

#### Εκμάθηση
- **UC1**: Μελέτη Θεωρίας
- **UC2**: Προβολή Ενοτήτων AI
- **UC3**: Ανάγνωση Εξηγήσεων
- **UC4**: Κατανόηση Εννοιών

#### Πρακτική Εξάσκηση
- **UC5**: Εκτέλεση Python Examples
- **UC6**: Διαδραστικές Εξομοιώσεις
- **UC7**: Πειραματισμός με Parameters
- **UC8**: Οπτικοποίηση Αποτελεσμάτων

#### Αξιολόγηση
- **UC9**: Απάντηση σε Quiz
- **UC10**: Αυτοαξιολόγηση
- **UC11**: Επίλυση Ασκήσεων

#### AI Βοηθός
- **UC12**: Υποβολή Ερωτήσεων στο Chatbot
- **UC13**: Λήψη Απαντήσεων
- **UC14**: Αναζήτηση στο Knowledge Base
- **UC15**: Web Search

#### Πόροι
- **UC16**: Πρόσβαση σε Colab Notebooks
- **UC17**: Προβολή Documentation
- **UC18**: Εξωτερικές Πηγές

### 🔗 Σχέσεις

**Include Relationships:**
- UC1 includes UC2, UC3
- UC5 includes UC7, UC8
- UC12 includes UC13
- UC13 includes UC14

**Extend Relationships:**
- UC2 extends UC4
- UC14 extends UC15

### 📝 PlantUML Source

```plantuml
diagrams/use_case_diagram.puml
```

### 💡 Σημαντικά Σημεία

1. **Multiple User Types**: Διαφορετικοί actors με διαφορετικές ανάγκες
2. **Comprehensive Coverage**: Κάλυψη όλων των λειτουργιών
3. **External Integration**: Σύνδεση με Wikipedia & Web
4. **Flexibility**: Πολλαπλοί τρόποι μάθησης

---

## Deployment Diagram - Deployment Architecture

### 📖 Περιγραφή

Το **Deployment Diagram** δείχνει την **αρχιτεκτονική ανάπτυξης** του συστήματος.

### 🎯 Deployment Nodes

#### 1. Client Device
- **Web Browser**: Chrome, Firefox, Safari, Edge
- **Connection**: HTTPS to server

#### 2. Streamlit Cloud (Production)
- **Streamlit Server**: Production server
- **Main App**: `ai_training_app.py`
- **Chatbot Module**: `chatbot.py`
- **Auto-deployment**: From GitHub

#### 3. Local Development Environment
- **Python 3.8+**: Runtime
- **Streamlit Local**: Development server
- **Source Code**: Local files
- **Port**: `http://localhost:8501`

#### 4. Data Storage
- **pdf_content.txt**: 731KB knowledge base
- **sample_data.csv**: Sample datasets
- **Location**: In repository

#### 5. External APIs
- **Wikipedia API**: REST API
- **Web Search**: HTTP requests

#### 6. Version Control
- **GitHub Repository**: `krimits/ai_training_app`
- **Branches**: main, feature branches
- **Artifacts**: Code, Docs, Diagrams

### 📊 Deployment Options

```
Production:
├─ Streamlit Cloud (Recommended)
├─ Heroku
├─ AWS/Azure/GCP
└─ Docker Container

Development:
└─ Local (localhost:8501)
```

### 🚀 Deployment Process

```
Local Development
    ↓ (git push)
GitHub Repository
    ↓ (auto-deploy)
Streamlit Cloud
    ↓ (HTTPS)
Client Browser
```

### 📝 PlantUML Source

```plantuml
diagrams/deployment_diagram.puml
```

### 💡 Σημαντικά Σημεία

1. **Cloud-Ready**: Έτοιμο για cloud deployment
2. **Version Control**: Git-based workflow
3. **Simple Deployment**: Streamlit Cloud auto-deploy
4. **Development/Production Parity**: Ίδιο environment

---

## Τεχνική Υλοποίηση

### 📦 Τεχνολογίες

#### Core
- **Python**: 3.8+
- **Streamlit**: 1.28.0+
- **Git**: Version control

#### ML/Data Science
- **scikit-learn**: ML algorithms
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization

#### Web/APIs
- **Requests**: HTTP library
- **Wikipedia API**: Knowledge retrieval

### 📁 Δομή Project

```
ai_training_app/
├── ai_training_app.py          # Main app (3107 lines)
├── chatbot.py                   # AI chatbot (1710 lines)
├── pdf_content.txt              # Knowledge base (731KB)
├── sample_data.csv              # Sample data
├── requirements.txt             # Dependencies
├── diagrams/                    # UML diagrams (NEW)
│   ├── class_diagram_chatbot.puml
│   ├── sequence_diagram_chatbot.puml
│   ├── component_diagram_architecture.puml
│   ├── activity_diagram_main_flow.puml
│   ├── use_case_diagram.puml
│   └── deployment_diagram.puml
└── documentation/               # Documentation
    ├── README.md
    ├── ARCHITECTURE.md (αυτό το αρχείο)
    ├── PROJECT_SUMMARY.md
    ├── CHATBOT_DOCS.md
    └── ...
```

### 🎨 Design Patterns

#### 1. Singleton Pattern (implicitly)
- Streamlit session state
- Chatbot instance

#### 2. Strategy Pattern
- Multiple search strategies (QA, KB, Wikipedia, Web)

#### 3. Factory Pattern
- Dynamic content generation
- Quiz/exercise creation

#### 4. Observer Pattern (Streamlit)
- UI updates on state changes
- Reactive programming

### 🔒 Ασφάλεια

- ✅ Δεν αποθηκεύονται προσωπικά δεδομένα
- ✅ Local execution (no server-side storage)
- ✅ Public APIs only (Wikipedia)
- ✅ Open source (MIT License)

### ⚡ Performance

- **Caching**: Streamlit `@st.cache_data` για ML models
- **Lazy Loading**: Content loaded on-demand
- **Efficient Search**: Keyword matching before full search

---

## 📊 Σύνοψη UML Διαγραμμάτων

| Διάγραμμα | Σκοπός | Κύριες Πληροφορίες |
|-----------|--------|-------------------|
| **Class** | Δομή κλάσεων | AIKnowledgeBot structure, methods, dependencies |
| **Sequence** | Χρονική ροή | Chatbot interaction flow, fallback mechanism |
| **Component** | Αρχιτεκτονική | Layers, modules, external services |
| **Activity** | Ροή διεργασιών | User journey, tab navigation, workflows |
| **Use Case** | Λειτουργικότητα | User roles, features, requirements |
| **Deployment** | Ανάπτυξη | Production/dev environments, deployment process |

---

## 🔗 Πρόσθετοι Πόροι

### Documentation
- [README.md](../README.md) - Γενική περιγραφή
- [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md) - Ολοκληρωμένη σύνοψη
- [CHATBOT_DOCS.md](../CHATBOT_DOCS.md) - Chatbot documentation
- [COLAB_NOTEBOOKS.md](../COLAB_NOTEBOOKS.md) - Colab guides

### PlantUML Resources
- [PlantUML Official Site](https://plantuml.com/)
- [PlantUML Language Reference](https://plantuml.com/guide)
- [Real World PlantUML](https://real-world-plantuml.com/)

### UML Resources
- [UML 2.5 Specification](https://www.omg.org/spec/UML/)
- [UML Best Practices](https://www.uml-diagrams.org/)

---

## 📝 Συντήρηση Διαγραμμάτων

### Πότε να Ενημερώσετε τα Διαγράμματα:

1. **Class Diagram**: Όταν προστίθενται/αλλάζουν κλάσεις ή μέθοδοι
2. **Sequence Diagram**: Όταν αλλάζει η ροή αλληλεπίδρασης
3. **Component Diagram**: Όταν προστίθενται νέα modules/components
4. **Activity Diagram**: Όταν αλλάζει η ροή της εφαρμογής
5. **Use Case Diagram**: Όταν προστίθενται νέες λειτουργίες
6. **Deployment Diagram**: Όταν αλλάζει το deployment setup

### Οδηγίες Ενημέρωσης:

1. Επεξεργαστείτε το αντίστοιχο `.puml` αρχείο
2. Ελέγξτε το διάγραμμα με PlantUML viewer
3. Ενημερώστε αυτό το documentation αν χρειάζεται
4. Commit με περιγραφικό message

---

<div align="center">

## 🎓 AI Training App Architecture

**Version**: 2.0.0  
**Last Updated**: Νοέμβριος 2025  
**Maintained by**: [@krimits](https://github.com/krimits)

---

**Made with ❤️ for Education**

*"Good architecture is not about the perfect design, but about the right design for the problem at hand"*

</div>
