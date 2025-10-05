import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time

st.set_page_config(layout='wide', page_title='AI Training — Πλήρης Εκπαίδευση', page_icon='🤖')

# Helper function για διαδραστικά concept explanations
def concept_explainer(term, definition, details="", examples=""):
    """Δημιουργεί ένα διαδραστικό expander για εξήγηση όρων"""
    with st.expander(f"ℹ️ **{term}** - Κάντε κλικ για περισσότερα"):
        st.markdown(f"**📖 Ορισμός:**\n{definition}")
        if details:
            st.markdown(f"\n**🔍 Λεπτομέρειες:**\n{details}")
        if examples:
            st.markdown(f"\n**💡 Παραδείγματα:**\n{examples}")

# Helper function για Google Colab links
def colab_button(notebook_name, colab_url, description=""):
    """Δημιουργεί κουμπί για άνοιγμα Google Colab notebook"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**📓 {notebook_name}**")
        if description:
            st.caption(description)
    with col2:
        st.markdown(f"""
        <a href="{colab_url}" target="_blank">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
        </a>
        """, unsafe_allow_html=True)

def section_title(t): 
    st.markdown(f'## 📌 {t}')

def subsection_title(t):
    st.markdown(f'### {t}')

def show_quiz(q):
    st.write('**Ερώτηση:**', q['question'])
    if q['type'] == 'mcq':
        choice = st.radio('Επιλογές:', q['options'], key=q['id'], label_visibility='visible')
        if st.button('Υποβολή απάντησης', key=q['id']+'_btn'):
            if choice == q['answer']:
                st.success('✅ Σωστό! ' + q.get('explain',''))
                st.balloons()
            else:
                st.error('❌ Λάθος. ' + q.get('explain',''))
    elif q['type'] == 'tf':
        choice = st.radio('Επιλέξτε:', ['Σωστό','Λάθος'], key=q['id'], label_visibility='visible')
        if st.button('Υποβολή', key=q['id']+'_btn'):
            ans = 'Σωστό' if q['answer']==True else 'Λάθος'
            if choice==ans:
                st.success('✅ Σωστό! ' + q.get('explain',''))
                st.balloons()
            else:
                st.error('❌ Λάθος. ' + q.get('explain',''))

st.title('🤖 AI Training — Πλήρης Εκπαίδευση στην Τεχνητή Νοημοσύνη')
st.markdown('### Εφαρμογές Τεχνητής Νοημοσύνης και ChatGPT σε Κρίσιμους Τομείς')
st.markdown('---')

tabs = st.tabs(['📚 Περιεχόμενο','🐍 Παραδείγματα Python','🔬 Εξομοιώσεις AI','✅ Κουίζ','💡 Διαδραστικές Ασκήσεις','🤖 AI Chatbot','📖 Πόροι'])

with tabs[0]:
    section_title('1.1 Εισαγωγή — Τι είναι η Τεχνητή Νοημοσύνη')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        #### Ορισμός και Γενικά
        Η **Τεχνητή Νοημοσύνη (Artificial Intelligence - AI)** είναι ο κλάδος της επιστήμης των υπολογιστών 
        που ασχολείται με τη δημιουργία συστημάτων ικανών να εκτελούν εργασίες που παραδοσιακά απαιτούν 
        ανθρώπινη νοημοσύνη.
        
        **Βασικές Κατηγορίες:**
        """)
        
    with col2:
        st.info("""
        **💡 Γνωρίζατε ότι:**
        
        Η AI χρησιμοποιείται σε:
        - Smartphones (Siri, Alexa)
        - Αυτόνομα οχήματα
        - Ιατρικές διαγνώσεις
        - Οικονομικές προβλέψεις
        - Εξυπηρέτηση πελατών
        """)
    
    # Expandable sections for each AI category
    with st.expander('🧠 **Machine Learning (ML)** - Μηχανές που μαθαίνουν από δεδομένα', expanded=False):
        st.markdown("""
        ### Τι είναι το Machine Learning;
        
        Το **Machine Learning** είναι ένας κλάδος της AI που επιτρέπει στους υπολογιστές να μαθαίνουν από δεδομένα 
        χωρίς να προγραμματίζονται ρητά για κάθε εργασία.
        
        ---
        """)
        
        st.markdown("#### 📚 Τύποι Μάθησης")
        st.markdown("*Κάντε κλικ σε κάθε τύπο για αναλυτική εξήγηση:*")
        
        # SUPERVISED LEARNING
        concept_explainer(
            "🎯 Supervised Learning (Επιβλεπόμενη Μάθηση)",
            """
            Η **Supervised Learning** είναι η πιο συνηθισμένη κατηγορία ML όπου το μοντέλο μαθαίνει από 
            **labeled data** (δεδομένα με ετικέτες). Κάθε παράδειγμα εκπαίδευσης έχει input features και 
            το αντίστοιχο output (label/target).
            """,
            """
            ### 🎓 Πώς Λειτουργεί:
            
            **Βήμα 1**: Παρέχουμε στο μοντέλο ζεύγη (input, output)
            ```
            Παράδειγμα: (Σπίτι 100τμ με 3 δωμάτια) → 200,000€
            ```
            
            **Βήμα 2**: Το μοντέλο μαθαίνει τη σχέση input-output
            
            **Βήμα 3**: Προβλέπει outputs για νέα, άγνωστα inputs
            
            ---
            
            ### 📊 Δύο Κύριοι Τύποι:
            
            #### 1️⃣ **Classification (Ταξινόμηση)**
            - **Στόχος**: Πρόβλεψη διακριτής κατηγορίας
            - **Output**: Κατηγορία/Class (π.χ. "Σκύλος", "Γάτα")
            - **Παραδείγματα**:
              - Email spam detection (Spam/Not Spam)
              - Medical diagnosis (Υγιής/Άρρωστος)
              - Sentiment analysis (Positive/Negative/Neutral)
              - Face recognition (Πρόσωπο A, B, C...)
            
            **Αλγόριθμοι Classification:**
            - **Logistic Regression**: Για binary classification
            - **Decision Trees**: Tree-based decisions
            - **Random Forest**: Ensemble of trees
            - **Support Vector Machines (SVM)**: Finds optimal boundary
            - **Neural Networks**: Multi-layer learning
            - **Naive Bayes**: Probabilistic classifier
            - **K-Nearest Neighbors (KNN)**: Distance-based
            
            #### 2️⃣ **Regression (Παλινδρόμηση)**
            - **Στόχος**: Πρόβλεψη συνεχούς αριθμητικής τιμής
            - **Output**: Αριθμός (π.χ. 250,000€, 25 χρονών)
            - **Παραδείγματα**:
              - House price prediction
              - Stock market forecasting
              - Temperature prediction
              - Sales forecasting
              - Age estimation from photos
            
            **Αλγόριθμοι Regression:**
            - **Linear Regression**: Γραμμική σχέση
            - **Polynomial Regression**: Μη-γραμμική σχέση
            - **Ridge/Lasso Regression**: Με regularization
            - **Decision Tree Regression**: Tree-based
            - **Random Forest Regression**: Ensemble
            - **Support Vector Regression (SVR)**
            - **Neural Network Regression**
            
            ---
            
            ### ⚙️ Βασικά Στοιχεία:
            
            **Training Data (Εκπαιδευτικά Δεδομένα):**
            - X (features/inputs): Χαρακτηριστικά
            - y (labels/targets): Στόχοι/Ετικέτες
            - Παράδειγμα: X = [μέγεθος, δωμάτια, τοποθεσία], y = [τιμή]
            
            **Loss Function (Συνάρτηση Κόστους):**
            - Μετρά πόσο κοντά είναι οι προβλέψεις στην πραγματικότητα
            - Classification: Cross-Entropy Loss
            - Regression: Mean Squared Error (MSE)
            
            **Optimization (Βελτιστοποίηση):**
            - **Gradient Descent**: Βρίσκει minimum της loss function
            - Ενημερώνει παραμέτρους για καλύτερες προβλέψεις
            
            ---
            
            ### 📈 Μετρικές Αξιολόγησης:
            
            **Για Classification:**
            - **Accuracy**: Ποσοστό σωστών προβλέψεων
            - **Precision**: Από όσα προβλέψαμε θετικά, πόσα ήταν σωστά
            - **Recall**: Από όσα είναι θετικά, πόσα βρήκαμε
            - **F1-Score**: Αρμονικός μέσος Precision & Recall
            - **Confusion Matrix**: Πίνακας σωστών/λάθος προβλέψεων
            - **ROC-AUC**: Area Under the ROC Curve
            
            **Για Regression:**
            - **MAE** (Mean Absolute Error): Μέσο απόλυτο λάθος
            - **MSE** (Mean Squared Error): Μέσο τετραγωνικό λάθος
            - **RMSE** (Root MSE): Ρίζα του MSE
            - **R² Score**: Πόσο καλά εξηγείται η διακύμανση
            
            ---
            
            ### 💼 Real-World Applications:
            
            **Business:**
            - Customer churn prediction
            - Lead scoring
            - Price optimization
            - Demand forecasting
            
            **Healthcare:**
            - Disease diagnosis από symptoms
            - Patient risk stratification
            - Drug response prediction
            
            **Finance:**
            - Credit scoring
            - Fraud detection
            - Stock price prediction
            - Loan default prediction
            
            **E-commerce:**
            - Product recommendations
            - Dynamic pricing
            - Inventory management
            
            ---
            
            ### ⚠️ Προκλήσεις:
            
            - **Labeled Data Requirement**: Χρειάζεται πολλά labeled examples (ακριβό!)
            - **Overfitting**: Μαθαίνει το noise αντί για patterns
            - **Class Imbalance**: Άνισες κατηγορίες (π.χ. 99% κανονικά, 1% απάτη)
            - **Feature Engineering**: Επιλογή σωστών features είναι κρίσιμη
            """,
            """
            **Πότε να χρησιμοποιήσετε:**
            - ✅ Έχετε labeled data
            - ✅ Ξέρετε τι θέλετε να προβλέψετε (clear target)
            - ✅ Θέλετε interpretable results
            - ✅ Έχετε αρκετά δεδομένα για training
            
            **Παράδειγμα Code (Python):**
            ```python
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Train model
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            
            # Predict
            predictions = model.predict(X_test)
            accuracy = model.score(X_test, y_test)
            ```
            """
        )
        
        # UNSUPERVISED LEARNING
        concept_explainer(
            "🔍 Unsupervised Learning (Μη Επιβλεπόμενη Μάθηση)",
            """
            Η **Unsupervised Learning** μαθαίνει από **unlabeled data** (χωρίς ετικέτες). Το μοντέλο 
            προσπαθεί να ανακαλύψει κρυφά patterns, δομές και σχέσεις στα δεδομένα αυτόνομα.
            """,
            """
            ### 🎓 Πώς Λειτουργεί:
            
            **Δεν έχουμε labels** - Μόνο inputs!
            ```
            Παράδειγμα: Έχουμε 10,000 φωτογραφίες αλλά δεν ξέρουμε τι δείχνουν
            ```
            
            Το μοντέλο **αυτόνομα** ανακαλύπτει:
            - Ομάδες παρόμοιων δεδομένων (clusters)
            - Κρυφές διαστάσεις
            - Anomalies (ασυνήθιστα patterns)
            
            ---
            
            ### 📊 Κύριοι Τύποι:
            
            #### 1️⃣ **Clustering (Ομαδοποίηση)**
            - **Στόχος**: Ομαδοποίηση παρόμοιων δεδομένων
            - **Output**: Cluster assignments
            
            **Αλγόριθμοι Clustering:**
            
            **K-Means:**
            - Το πιο δημοφιλές
            - Χωρίζει δεδομένα σε K clusters
            - Γρήγορο και αποδοτικό
            - Παράδειγμα: Ομαδοποίηση πελατών
            
            **Hierarchical Clustering:**
            - Δημιουργεί dendrogram (δέντρο)
            - Δεν χρειάζεται να ορίσεις K εκ των προτέρων
            - Παράδειγμα: Ταξινόμηση ειδών (biology)
            
            **DBSCAN:**
            - Density-based clustering
            - Βρίσκει clusters αυθαίρετου σχήματος
            - Ανιχνεύει outliers
            - Παράδειγμα: Γεωγραφική ανάλυση
            
            **Gaussian Mixture Models (GMM):**
            - Probabilistic clustering
            - Soft assignments (πιθανότητες)
            
            **Εφαρμογές Clustering:**
            - Customer segmentation (marketing)
            - Image segmentation
            - Document clustering
            - Anomaly detection
            - Gene expression analysis
            
            #### 2️⃣ **Dimensionality Reduction (Μείωση Διαστάσεων)**
            - **Στόχος**: Μείωση αριθμού features διατηρώντας πληροφορία
            - **Output**: Μειωμένες διαστάσεις
            
            **Αλγόριθμοι:**
            
            **PCA (Principal Component Analysis):**
            - Βρίσκει principal components (κύριες διευθύνσεις διακύμανσης)
            - Linear transformation
            - Παράδειγμα: Από 1000 features → 50 features
            
            **t-SNE (t-Distributed Stochastic Neighbor Embedding):**
            - Για visualization (2D/3D)
            - Διατηρεί local structure
            - Αργό για μεγάλα datasets
            
            **UMAP (Uniform Manifold Approximation and Projection):**
            - Πιο γρήγορο από t-SNE
            - Καλύτερο για μεγάλα datasets
            
            **Autoencoders:**
            - Neural network-based
            - Μαθαίνει compressed representation
            
            **Εφαρμογές:**
            - Feature extraction
            - Data visualization
            - Noise reduction
            - Compression
            
            #### 3️⃣ **Association Rule Learning**
            - **Στόχος**: Βρες σχέσεις μεταξύ items
            - **Output**: Rules (IF...THEN)
            
            **Αλγόριθμοι:**
            - **Apriori**: Market basket analysis
            - **FP-Growth**: Faster than Apriori
            
            **Παράδειγμα:**
            ```
            IF (αγοράζει ψωμί AND αγοράζει βούτυρο) 
            THEN (πιθανόν να αγοράσει και μαρμελάδα)
            ```
            
            **Εφαρμογές:**
            - Market basket analysis (e-commerce)
            - Recommendation systems
            - Web usage mining
            
            #### 4️⃣ **Anomaly Detection (Ανίχνευση Ανωμαλιών)**
            - **Στόχος**: Βρες ασυνήθιστα/outlier data points
            
            **Αλγόριθμοι:**
            - **Isolation Forest**
            - **One-Class SVM**
            - **Local Outlier Factor (LOF)**
            
            **Εφαρμογές:**
            - Fraud detection
            - System health monitoring
            - Quality control
            
            ---
            
            ### 📈 Αξιολόγηση:
            
            **Clustering Metrics:**
            - **Silhouette Score**: Πόσο καλά χωρίζονται τα clusters (-1 to 1)
            - **Davies-Bouldin Index**: Μικρότερο = καλύτερα
            - **Calinski-Harabasz Score**: Μεγαλύτερο = καλύτερα
            - **Inertia**: Εσωτερική διασπορά clusters
            
            **Dimensionality Reduction:**
            - **Explained Variance Ratio**: Πόση πληροφορία διατηρείται
            - **Reconstruction Error**: Πόσο καλά μπορούμε να ανακατασκευάσουμε τα original data
            
            ---
            
            ### 💼 Real-World Applications:
            
            **Marketing:**
            - Customer segmentation (ομαδοποίηση πελατών σε segments)
            - Market basket analysis (τι αγοράζουν μαζί)
            
            **Healthcare:**
            - Patient stratification
            - Disease subtype discovery
            - Gene expression analysis
            
            **Finance:**
            - Fraud detection (ανωμαλίες σε συναλλαγές)
            - Portfolio optimization
            
            **Social Media:**
            - Community detection
            - Topic modeling
            - Trend analysis
            
            **Manufacturing:**
            - Defect detection
            - Process monitoring
            
            ---
            
            ### ⚠️ Προκλήσεις:
            
            - **Evaluation is Tricky**: Δύσκολο να αξιολογήσεις χωρίς ground truth
            - **Parameter Tuning**: (π.χ. πόσα clusters να διαλέξεις;)
            - **Interpretability**: Τι σημαίνει το κάθε cluster;
            - **Scalability**: Μερικοί αλγόριθμοι αργοί για big data
            """,
            """
            **Πότε να χρησιμοποιήσετε:**
            - ✅ ΔΕΝ έχετε labels (ή είναι ακριβό να τα φτιάξετε)
            - ✅ Θέλετε exploratory analysis
            - ✅ Ψάχνετε hidden patterns
            - ✅ Preprocessing για supervised learning
            
            **Παράδειγμα Code (K-Means):**
            ```python
            from sklearn.cluster import KMeans
            import matplotlib.pyplot as plt
            
            # Fit K-Means
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            # Visualize
            plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
            plt.scatter(kmeans.cluster_centers_[:, 0], 
                       kmeans.cluster_centers_[:, 1], 
                       c='red', marker='X', s=200)
            plt.show()
            ```
            
            **Παράδειγμα Code (PCA):**
            ```python
            from sklearn.decomposition import PCA
            
            # Reduce from 100 features to 10
            pca = PCA(n_components=10)
            X_reduced = pca.fit_transform(X)
            
            # Check explained variance
            print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
            ```
            """
        )
        
        # REINFORCEMENT LEARNING
        concept_explainer(
            "🎮 Reinforcement Learning (Ενισχυτική Μάθηση)",
            """
            Η **Reinforcement Learning** είναι ένα παράδειγμα μάθησης όπου ένας **agent** (πράκτορας) 
            μαθαίνει να παίρνει αποφάσεις αλληλεπιδρώντας με ένα **environment** (περιβάλλον) μέσω δοκιμής 
            και λάθους, με στόχο τη **μεγιστοποίηση των rewards** (ανταμοιβών).
            """,
            """
            ### 🎓 Πώς Λειτουργεί:
            
            **Βασική Ιδέα**: Trial and Error + Rewards
            
            ```
            Agent (π.χ. ρομπότ) → Action → Environment
                                        ↓
                             State & Reward ← Environment
                                        ↓
                            Agent μαθαίνει τι είναι καλό/κακό
            ```
            
            **Διαφορά από Supervised:**
            - Supervised: "Αυτό είναι το σωστό" (explicit labels)
            - RL: "Αυτό είναι καλό/κακό" (rewards/penalties)
            
            ---
            
            ### 🧩 Βασικά Στοιχεία:
            
            #### 1. **Agent (Πράκτορας)**
            - Το "όν" που μαθαίνει και παίρνει αποφάσεις
            - Παράδειγμα: Ρομπότ, AI παίκτης, trading bot
            
            #### 2. **Environment (Περιβάλλον)**
            - Ο κόσμος με τον οποίο αλληλεπιδρά ο agent
            - Παράδειγμα: Σκακιέρα, παιχνίδι video game, χρηματιστήριο
            
            #### 3. **State (Κατάσταση)**
            - Η τρέχουσα κατάσταση του περιβάλλοντος
            - Παράδειγμα: Θέσεις κομματιών στο σκάκι
            
            #### 4. **Action (Ενέργεια)**
            - Τι μπορεί να κάνει ο agent
            - Παράδειγμα: Μετακίνηση πιονιού, πήδημα, αγορά μετοχής
            
            #### 5. **Reward (Ανταμοιβή)**
            - Αριθμητικό signal που λέει πόσο καλή ήταν η action
            - Positive: Καλό (+1, +10, +100)
            - Negative: Κακό (-1, -10, -100)
            - Zero: Ουδέτερο
            
            #### 6. **Policy (Πολιτική) π**
            - Η στρατηγική του agent: State → Action
            - "Τι action να κάνω σε κάθε state"
            
            #### 7. **Value Function V(s)**
            - Πόσο "καλό" είναι ένα state μακροπρόθεσμα
            - Λαμβάνει υπόψη μελλοντικά rewards
            
            #### 8. **Q-Function Q(s,a)**
            - Πόσο "καλή" είναι μια action σε ένα state
            - Q(state, action) = Expected future reward
            
            ---
            
            ### 🎯 Τύποι RL:
            
            #### **Model-Free RL** (Δεν έχει μοντέλο του environment)
            - Μαθαίνει απευθείας από experience
            - Πιο κοινό στην πράξη
            
            #### **Model-Based RL** (Έχει μοντέλο του environment)
            - Μαθαίνει πώς λειτουργεί το environment
            - Μπορεί να σχεδιάζει μπροστά (planning)
            
            ---
            
            ### ⚙️ Κύριοι Αλγόριθμοι:
            
            #### 1️⃣ **Q-Learning** (Value-Based)
            
            **Ιδέα**: Μάθε την Q-function (quality of actions)
            
            **Update Rule:**
            ```
            Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
            ```
            
            Όπου:
            - α: Learning rate (πόσο γρήγορα μαθαίνει)
            - γ (gamma): Discount factor (πόσο σημαντικά είναι future rewards)
            - r: Reward που έλαβε
            - s': Next state
            
            **Χαρακτηριστικά:**
            - Off-policy (μπορεί να μάθει από άλλους agents)
            - Convergence guaranteed (υπό προϋποθέσεις)
            - Κλασικός αλγόριθμος
            
            **Εφαρμογές:**
            - Grid world navigation
            - Simple games
            
            #### 2️⃣ **Deep Q-Networks (DQN)** (Deep RL)
            
            **Ιδέα**: Χρήση Neural Network για Q-function
            
            **Innovations:**
            - **Experience Replay**: Αποθηκεύει (s,a,r,s') και τα replay
            - **Target Network**: Σταθεροποίηση training
            
            **Επιτεύγματα:**
            - DeepMind's Atari games (2013)
            - Superhuman performance σε πολλά games
            
            #### 3️⃣ **Policy Gradients** (Policy-Based)
            
            **Ιδέα**: Μάθε απευθείας την policy π(a|s)
            
            **Αλγόριθμοι:**
            - **REINFORCE**: Βασικός policy gradient
            - **Actor-Critic**: Συνδυασμός value + policy
            - **A3C** (Asynchronous Advantage Actor-Critic): Παράλληλοι agents
            - **PPO** (Proximal Policy Optimization): State-of-the-art, stable
            - **TRPO** (Trust Region Policy Optimization)
            
            **Πλεονεκτήματα:**
            - Δουλεύει σε continuous action spaces
            - Stochastic policies (πιθανοτικές)
            
            #### 4️⃣ **Advanced Algorithms**
            
            **DDPG** (Deep Deterministic Policy Gradient):
            - Για continuous control
            - Ρομποτική manipulation
            
            **SAC** (Soft Actor-Critic):
            - Maximum entropy RL
            - Πολύ stable
            
            **AlphaGo/AlphaZero:**
            - Monte Carlo Tree Search + Deep RL
            - Superhuman Go playing
            
            ---
            
            ### 📈 Exploration vs Exploitation:
            
            **Δίλημμα**: Να δοκιμάσω νέα (explore) ή να κάνω το γνωστό καλό (exploit);
            
            **Strategies:**
            - **ε-greedy**: Με πιθανότητα ε κάνε random action
            - **Softmax**: Probabilistic selection
            - **Upper Confidence Bound (UCB)**: Optimistic exploration
            
            ---
            
            ### 💼 Real-World Applications:
            
            **Gaming:**
            - AlphaGo (Go)
            - OpenAI Five (Dota 2)
            - AlphaStar (StarCraft II)
            - Game AI characters
            
            **Robotics:**
            - Robotic manipulation (πιάσιμο αντικειμένων)
            - Locomotion (περπάτημα)
            - Autonomous navigation
            - Assembly tasks
            
            **Autonomous Vehicles:**
            - Path planning
            - Adaptive cruise control
            - Parking
            
            **Finance:**
            - Trading strategies
            - Portfolio management
            - Dynamic pricing
            
            **Healthcare:**
            - Treatment optimization
            - Personalized medicine
            - Drug dosing
            
            **Recommender Systems:**
            - YouTube recommendations
            - News feed optimization
            - Ad placement
            
            **Energy:**
            - Data center cooling (Google)
            - Smart grid management
            - Building HVAC control
            
            **Natural Language:**
            - Dialogue systems
            - Neural architecture search
            - Machine translation improvements
            
            ---
            
            ### 📊 Challenges:
            
            #### **Sample Efficiency**
            - Χρειάζεται ΠΟΛΛΑ δείγματα (millions!)
            - Αργή εκπαίδευση
            - Λύση: Transfer learning, sim-to-real
            
            #### **Reward Engineering**
            - Δύσκολο να ορίσεις σωστά rewards
            - Reward hacking (agent βρίσκει shortcuts)
            - Λύση: Inverse RL, reward shaping
            
            #### **Stability**
            - Training μπορεί να diverge
            - Sensitive σε hyperparameters
            - Λύση: PPO, SAC (πιο stable algorithms)
            
            #### **Credit Assignment**
            - Ποια action ήταν υπεύθυνη για το reward;
            - Temporal credit assignment problem
            
            #### **Exploration**
            - Πώς να explore αποδοτικά;
            - Sparse rewards (λίγα rewards)
            
            ---
            
            ### 🛠️ Frameworks & Tools:
            
            **OpenAI Gym:**
            - Standard RL environments
            - Atari, MuJoCo, Robotics
            
            **Stable Baselines3:**
            - Reliable RL implementations
            - PPO, A2C, SAC, TD3, DQN
            
            **Ray RLlib:**
            - Scalable RL
            - Distributed training
            
            **TF-Agents:**
            - TensorFlow RL library
            
            **PyTorch RL:**
            - Πολλές implementations
            """,
            """
            **Πότε να χρησιμοποιήσετε:**
            - ✅ Έχετε sequential decision problem
            - ✅ Μπορείτε να ορίσετε rewards
            - ✅ Έχετε simulator (ή real environment)
            - ✅ Χρειάζεστε adaptive behavior
            - ✅ Το problem έχει long-term consequences
            
            **Παράδειγμα Code (Q-Learning):**
            ```python
            import numpy as np
            
            # Initialize Q-table
            Q = np.zeros([n_states, n_actions])
            alpha = 0.1  # learning rate
            gamma = 0.99  # discount factor
            
            for episode in range(1000):
                state = env.reset()
                done = False
                
                while not done:
                    # ε-greedy action selection
                    if np.random.random() < epsilon:
                        action = env.action_space.sample()  # explore
                    else:
                        action = np.argmax(Q[state, :])  # exploit
                    
                    # Take action
                    next_state, reward, done, _ = env.step(action)
                    
                    # Q-learning update
                    Q[state, action] = Q[state, action] + alpha * (
                        reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
                    )
                    
                    state = next_state
            ```
            
            **Παράδειγμα Code (PPO με Stable-Baselines3):**
            ```python
            from stable_baselines3 import PPO
            import gym
            
            # Create environment
            env = gym.make('CartPole-v1')
            
            # Train PPO agent
            model = PPO('MlpPolicy', env, verbose=1)
            model.learn(total_timesteps=10000)
            
            # Test trained agent
            obs = env.reset()
            for i in range(1000):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
            ```
            """
        )
        
        st.markdown("---")
        st.markdown("""
        #### 🔍 Βασικά Στάδια ML Pipeline
        
        1. **Data Collection** (Συλλογή Δεδομένων)
           - Συγκέντρωση σχετικών δεδομένων
           - Ποιότητα > Ποσότητα
        
        2. **Data Preprocessing** (Προεπεξεργασία)
           - Καθαρισμός δεδομένων
           - Χειρισμός missing values
           - Normalization/Standardization
           - Feature Engineering
        
        3. **Model Selection** (Επιλογή Μοντέλου)
           - Επιλογή κατάλληλου αλγορίθμου
           - Υπερπαράμετροι (hyperparameters)
        
        4. **Training** (Εκπαίδευση)
           - Fit του μοντέλου στα training data
           - Optimization (π.χ. Gradient Descent)
        
        5. **Evaluation** (Αξιολόγηση)
           - Μετρικές: Accuracy, Precision, Recall, F1-Score
           - Cross-validation
           - Confusion Matrix
        
        6. **Deployment** (Παραγωγή)
           - Θέση σε production
           - Monitoring και maintenance
        
        #### 💼 Πρακτικές Εφαρμογές
        
        - **E-commerce**: Προτάσεις προϊόντων (Amazon, Netflix)
        - **Finance**: Credit scoring, fraud detection
        - **Healthcare**: Διάγνωση ασθενειών, drug discovery
        - **Marketing**: Customer segmentation, churn prediction
        - **Manufacturing**: Predictive maintenance, quality control
        
        #### 📊 Δημοφιλείς Βιβλιοθήκες Python
        
        - **scikit-learn**: Γενικού σκοπού ML
        - **XGBoost**: Gradient boosting
        - **LightGBM**: Fast gradient boosting
        - **CatBoost**: Categorical features handling
        
        #### ⚠️ Προκλήσεις
        
        - **Overfitting**: Το μοντέλο μαθαίνει "απ' έξω" τα training data
        - **Underfitting**: Το μοντέλο είναι πολύ απλό
        - **Bias in Data**: Μεροληψία στα δεδομένα
        - **Feature Engineering**: Δημιουργία σωστών features
        """)
    
    with st.expander('🌐 **Deep Learning** - Νευρωνικά δίκτυα με πολλά επίπεδα', expanded=False):
        st.markdown("""
        ### Τι είναι το Deep Learning;
        
        Το **Deep Learning** είναι υποκατηγορία του Machine Learning που χρησιμοποιεί **νευρωνικά δίκτυα** 
        με πολλά κρυφά επίπεδα (layers) για να μάθει πολύπλοκες αναπαραστάσεις από δεδομένα.
        
        #### 🧬 Αρχιτεκτονικές Neural Networks
        
        **1. Feedforward Neural Networks (FNN)**
        - Το πιο βασικό τύπο νευρωνικού δικτύου
        - Πληροφορία ρέει προς τα εμπρός (input → hidden → output)
        - Χρήση: Tabular data, απλή classification/regression
        
        **2. Convolutional Neural Networks (CNN)**
        - Ειδικευμένα για **εικόνες** και spatial data
        - Convolution layers εξάγουν features
        - Pooling layers μειώνουν διαστάσεις
        - Παραδείγματα: ResNet, VGG, Inception, EfficientNet
        - Εφαρμογές: 
          - Image classification
          - Object detection (YOLO, Faster R-CNN)
          - Face recognition
          - Medical imaging
        
        **3. Recurrent Neural Networks (RNN)**
        - Για **sequential data** (κείμενο, χρονοσειρές)
        - Έχουν "μνήμη" προηγούμενων states
        - Παραδείγματα: LSTM, GRU
        - Εφαρμογές:
          - Natural Language Processing
          - Speech recognition
          - Time series prediction
          - Music generation
        
        **4. Transformer Architecture**
        - **Επανάσταση στο NLP** (2017)
        - Self-attention mechanism
        - Παράλληλη επεξεργασία (γρηγορότερο από RNN)
        - Παραδείγματα: BERT, GPT, T5, Vision Transformer (ViT)
        - Εφαρμογές:
          - Language models (ChatGPT, Claude)
          - Machine translation
          - Text summarization
          - Question answering
        
        **5. Generative Adversarial Networks (GANs)**
        - Δύο δίκτυα "παλεύουν" (Generator vs Discriminator)
        - Δημιουργία ρεαλιστικών δεδομένων
        - Εφαρμογές:
          - Image generation (StyleGAN, BigGAN)
          - DeepFakes
          - Data augmentation
          - Art creation
        
        **6. Autoencoders**
        - Συμπίεση και αποσυμπίεση δεδομένων
        - Μάθηση latent representations
        - Τύποι: VAE (Variational Autoencoders)
        - Εφαρμογές:
          - Dimensionality reduction
          - Anomaly detection
          - Denoising
          - Image compression
        
        #### 🎯 Βασικά Concepts
        
        **Activation Functions:**
        - ReLU (Rectified Linear Unit): f(x) = max(0, x)
        - Sigmoid: f(x) = 1/(1+e^(-x))
        - Tanh: f(x) = (e^x - e^(-x))/(e^x + e^(-x))
        - Softmax: για classification
        
        **Optimization Algorithms:**
        - SGD (Stochastic Gradient Descent)
        - Adam (Adaptive Moment Estimation)
        - RMSprop
        - AdaGrad
        
        **Regularization Techniques:**
        - Dropout: Απενεργοποίηση τυχαίων neurons
        - L1/L2 Regularization
        - Batch Normalization
        - Early Stopping
        
        #### 💻 Frameworks
        
        - **TensorFlow**: Google's framework
        - **PyTorch**: Facebook's framework (ερευνητικό favorite)
        - **Keras**: High-level API (τώρα μέρος του TensorFlow)
        - **JAX**: High-performance computing
        
        #### 🚀 Cutting-Edge Applications
        
        - **Computer Vision**: Self-driving cars, medical imaging
        - **NLP**: ChatGPT, Google Translate, sentiment analysis
        - **Speech**: Siri, Alexa, speech-to-text
        - **Gaming**: AlphaGo, OpenAI Five (Dota 2)
        - **Science**: Protein folding (AlphaFold), drug discovery
        
        #### ⚡ Απαιτήσεις
        
        - **Hardware**: GPU/TPU (NVIDIA, Google Cloud)
        - **Data**: Μεγάλα datasets (χιλιάδες-εκατομμύρια δείγματα)
        - **Time**: Εκπαίδευση μπορεί να πάρει ώρες/μέρες
        - **Expertise**: Γνώση hyperparameters, architectures
        """)
    
    with st.expander('💬 **Natural Language Processing (NLP)** - Επεξεργασία Φυσικής Γλώσσας', expanded=False):
        st.markdown("""
        ### Τι είναι το NLP;
        
        Το **Natural Language Processing** είναι ο κλάδος της AI που ασχολείται με την αλληλεπίδραση 
        μεταξύ υπολογιστών και ανθρώπινης γλώσσας.
        
        #### 📝 Βασικές Εργασίες NLP
        
        **1. Text Classification**
        - Sentiment Analysis (ανάλυση συναισθήματος)
        - Spam Detection
        - Topic Classification
        - Intent Detection (chatbots)
        
        **2. Named Entity Recognition (NER)**
        - Εντοπισμός ονομάτων, τόπων, ημερομηνιών
        - Εξαγωγή πληροφορίας από κείμενο
        
        **3. Machine Translation**
        - Μετάφραση μεταξύ γλωσσών
        - Google Translate, DeepL
        - Neural Machine Translation (NMT)
        
        **4. Question Answering**
        - Απάντηση σε ερωτήσεις
        - Reading comprehension
        - ChatGPT, Bing Chat
        
        **5. Text Summarization**
        - Αυτόματη σύνοψη κειμένων
        - Extractive vs Abstractive
        
        **6. Text Generation**
        - Δημιουργία κειμένου
        - GPT models, content creation
        - Story writing, code generation
        
        **7. Speech Recognition**
        - Speech-to-text
        - Siri, Google Assistant, Alexa
        
        **8. Part-of-Speech Tagging**
        - Ταυτοποίηση μερών του λόγου
        - Σύνταξη και γραμματική ανάλυση
        
        #### 🔤 Βασικά Στάδια NLP Pipeline
        
        **1. Tokenization**
        - Διαχωρισμός κειμένου σε tokens (λέξεις, προτάσεις)
        - Word tokenization, sentence tokenization
        
        **2. Text Cleaning**
        - Lowercase conversion
        - Αφαίρεση σημείων στίξης
        - Αφαίρεση stop words (the, and, is...)
        
        **3. Stemming / Lemmatization**
        - Stemming: running → run (αφαίρεση καταλήξεων)
        - Lemmatization: better → good (γραμματική μορφή)
        
        **4. Feature Extraction**
        - **Bag of Words (BoW)**: Συχνότητα λέξεων
        - **TF-IDF**: Term Frequency - Inverse Document Frequency
        - **Word Embeddings**: Word2Vec, GloVe, FastText
        - **Contextualized Embeddings**: BERT, ELMo
        
        **5. Model Training**
        - Traditional ML: Naive Bayes, SVM, Random Forest
        - Deep Learning: RNN, LSTM, Transformers
        
        #### 🤖 Σύγχρονα NLP Models
        
        **Pre-trained Language Models:**
        
        - **BERT** (Bidirectional Encoder Representations from Transformers)
          - Κατανόηση context από δύο κατευθύνσεις
          - Fine-tuning για specific tasks
        
        - **GPT** (Generative Pre-trained Transformer)
          - GPT-3, GPT-4: Μοντέλα δημιουργίας κειμένου
          - ChatGPT: Conversational AI
        
        - **T5** (Text-to-Text Transfer Transformer)
          - Όλες οι εργασίες ως text-to-text
        
        - **RoBERTa**: Optimized BERT
        
        - **XLNet**: Permutation language modeling
        
        - **ELECTRA**: Efficient pre-training
        
        #### 🌍 Multilingual NLP
        
        - **mBERT**: Multilingual BERT
        - **XLM-R**: Cross-lingual modeling
        - Υποστήριξη 100+ γλωσσών
        
        #### 💼 Εφαρμογές
        
        - **Chatbots**: Εξυπηρέτηση πελατών 24/7
        - **Virtual Assistants**: Siri, Alexa, Google Assistant
        - **Content Moderation**: Φιλτράρισμα toxic content
        - **Email Filtering**: Spam detection
        - **Social Media**: Sentiment analysis, trend detection
        - **Healthcare**: Clinical notes analysis
        - **Legal**: Contract analysis, document review
        - **Finance**: News analysis, earnings calls
        
        #### 📚 Βιβλιοθήκες Python
        
        - **NLTK**: Natural Language Toolkit (traditional)
        - **spaCy**: Industrial-strength NLP
        - **Transformers (Hugging Face)**: Pre-trained models
        - **Gensim**: Topic modeling, word embeddings
        - **TextBlob**: Simple NLP tasks
        
        #### 🎯 Προκλήσεις
        
        - **Ambiguity**: Πολυσημία λέξεων
        - **Context**: Κατανόηση πλαισίου
        - **Sarcasm/Irony**: Δύσκολο να ανιχνευθεί
        - **Cultural Nuances**: Πολιτισμικές διαφορές
        - **Low-resource Languages**: Λίγα δεδομένα για κάποιες γλώσσες
        """)
    
    with st.expander('👁️ **Computer Vision** - Όραση Υπολογιστών', expanded=False):
        st.markdown("""
        ### Τι είναι η Computer Vision;
        
        Η **Computer Vision** είναι ο κλάδος της AI που επιτρέπει στους υπολογιστές να "βλέπουν" 
        και να κατανοούν το περιεχόμενο εικόνων και βίντεο.
        
        #### 🎯 Βασικές Εργασίες Computer Vision
        
        **1. Image Classification**
        - Ταξινόμηση εικόνων σε κατηγορίες
        - Παράδειγμα: Γάτα vs Σκύλος
        - Datasets: ImageNet (1000 κατηγορίες)
        - Models: ResNet, VGG, Inception, EfficientNet
        
        **2. Object Detection**
        - Εντοπισμός αντικειμένων σε εικόνα
        - Bounding boxes + classification
        - Real-time detection
        - Algorithms:
          - **YOLO** (You Only Look Once): Real-time
          - **Faster R-CNN**: High accuracy
          - **SSD**: Single Shot Detector
          - **RetinaNet**: Focal loss
        
        **3. Semantic Segmentation**
        - Ταξινόμηση κάθε pixel σε κατηγορία
        - Pixel-level understanding
        - Models: U-Net, DeepLab, SegNet
        - Εφαρμογές: Autonomous driving, medical imaging
        
        **4. Instance Segmentation**
        - Διαχωρισμός επιμέρους instances
        - Mask R-CNN
        - Συνδυασμός detection + segmentation
        
        **5. Face Recognition**
        - Αναγνώριση προσώπου
        - Face detection → Face alignment → Face recognition
        - Models: FaceNet, DeepFace, ArcFace
        - Εφαρμογές: Security, photo tagging, authentication
        
        **6. Pose Estimation**
        - Εντοπισμός keypoints του σώματος
        - Skeleton detection
        - Models: OpenPose, PoseNet
        - Εφαρμογές: Sports analysis, AR/VR, fitness apps
        
        **7. Image Generation**
        - Δημιουργία νέων εικόνων
        - GANs, Diffusion Models
        - Παραδείγματα:
          - **StyleGAN**: Ρεαλιστικά πρόσωπα
          - **DALL-E**: Text-to-image
          - **Stable Diffusion**: Open-source generation
          - **Midjourney**: Artistic images
        
        **8. Video Analysis**
        - Action recognition
        - Video classification
        - Tracking objects σε βίντεο
        - Activity detection
        
        #### 🏗️ Αρχιτεκτονικές CNN για Computer Vision
        
        **Classic Architectures:**
        - **LeNet-5** (1998): Πρώτο CNN για MNIST
        - **AlexNet** (2012): ImageNet winner, ReLU, dropout
        - **VGG** (2014): Πολύ βαθύ δίκτυο (16-19 layers)
        - **GoogleNet/Inception** (2014): Inception modules
        
        **Modern Architectures:**
        - **ResNet** (2015): Residual connections, 50-152 layers
        - **DenseNet** (2017): Dense connections
        - **EfficientNet** (2019): Optimal scaling
        - **Vision Transformer (ViT)** (2020): Transformers for images
        - **Swin Transformer** (2021): Hierarchical transformers
        
        #### 🔍 Βασικά Concepts
        
        **Convolution:**
        - Filters/Kernels εξάγουν features
        - Edge detection, texture, patterns
        - Spatial hierarchy (low → high level features)
        
        **Pooling:**
        - Max pooling, Average pooling
        - Μείωση διαστάσεων
        - Translation invariance
        
        **Data Augmentation:**
        - Rotation, flipping, cropping
        - Color jittering
        - Mixup, CutMix
        - Αύξηση dataset artificially
        
        **Transfer Learning:**
        - Pre-trained models σε ImageNet
        - Fine-tuning για specific task
        - Feature extraction
        
        #### 💼 Εφαρμογές στην Πράξη
        
        **Αυτόνομα Οχήματα:**
        - Lane detection
        - Object detection (πεζοί, οχήματα)
        - Traffic sign recognition
        - Depth estimation
        
        **Healthcare:**
        - X-ray analysis
        - CT/MRI scan interpretation
        - Skin cancer detection
        - Retinal disease diagnosis
        - COVID-19 detection
        
        **Retail:**
        - Visual search (εύρεση προϊόντων από φωτό)
        - Cashier-less stores (Amazon Go)
        - Inventory management
        
        **Security:**
        - Surveillance systems
        - Anomaly detection
        - Facial recognition για access control
        
        **Agriculture:**
        - Crop monitoring
        - Disease detection σε φυτά
        - Yield prediction
        
        **Manufacturing:**
        - Quality inspection
        - Defect detection
        - Assembly verification
        
        **Social Media:**
        - Auto-tagging φωτογραφιών
        - Content moderation
        - Filters και effects (Snapchat, Instagram)
        
        #### 🛠️ Tools και Frameworks
        
        **Deep Learning:**
        - TensorFlow, PyTorch
        - Keras
        - ONNX (model interchange)
        
        **Computer Vision Libraries:**
        - **OpenCV**: Traditional CV algorithms
        - **Pillow (PIL)**: Image processing
        - **scikit-image**: Image algorithms
        - **Detectron2**: Facebook's CV library
        - **MMDetection**: Toolbox for object detection
        
        **Pre-trained Models:**
        - **Torchvision**: PyTorch models
        - **Keras Applications**: TensorFlow models
        - **Timm**: PyTorch Image Models
        
        #### 📊 Μετρικές Αξιολόγησης
        
        - **Classification**: Accuracy, Precision, Recall, F1
        - **Object Detection**: mAP (mean Average Precision), IoU
        - **Segmentation**: Dice coefficient, IoU
        - **Image Generation**: FID (Fréchet Inception Distance), IS (Inception Score)
        
        #### 🚧 Προκλήσεις
        
        - **Lighting conditions**: Φωτισμός επηρεάζει ποιότητα
        - **Occlusions**: Αντικείμενα κρυμμένα
        - **Scale variance**: Αντικείμενα σε διαφορετικά μεγέθη
        - **Real-time processing**: Χρειάζεται ταχύτητα
        - **3D understanding**: Από 2D εικόνες
        """)
    
    with st.expander('🤖 **Robotics** - Ρομποτική και Αυτονομία', expanded=False):
        st.markdown("""
        ### Τι είναι η Robotics με AI;
        
        Η **Robotics** συνδυάζει AI, μηχανική, και φυσική για τη δημιουργία ρομπότ που μπορούν 
        να αλληλεπιδρούν με το φυσικό κόσμο και να εκτελούν εργασίες αυτόνομα.
        
        #### 🎯 Βασικά Πεδία AI στη Ρομποτική
        
        **1. Perception (Αντίληψη)**
        - **Computer Vision**: Κάμερες για αναγνώριση αντικειμένων
        - **Sensor Fusion**: Συνδυασμός δεδομένων από πολλαπλούς αισθητήρες
        - **Depth Sensing**: LiDAR, RGB-D κάμερες
        - **Object Recognition**: Τι υπάρχει στο περιβάλλον;
        - **Scene Understanding**: Κατανόηση πλαισίου
        
        **2. Localization & Mapping (Εντοπισμός & Χαρτογράφηση)**
        - **SLAM** (Simultaneous Localization and Mapping)
          - Δημιουργία χάρτη ενώ το ρομπότ κινείται
          - Εντοπισμός θέσης στον χάρτη
        - **GPS Navigation**: Outdoor εντοπισμός
        - **Visual Odometry**: Υπολογισμός κίνησης από εικόνες
        - **Sensor-based Localization**: IMU, wheel encoders
        
        **3. Motion Planning (Σχεδιασμός Κίνησης)**
        - **Path Planning**: Εύρεση διαδρομής από A σε B
        - **Trajectory Optimization**: Βέλτιστη τροχιά
        - **Obstacle Avoidance**: Αποφυγή εμποδίων
        - **Algorithms**:
          - A* (A-star): Graph search
          - RRT (Rapidly-exploring Random Trees)
          - Dijkstra
          - Dynamic Window Approach
        
        **4. Control (Έλεγχος)**
        - **PID Controllers**: Proportional-Integral-Derivative
        - **Model Predictive Control (MPC)**
        - **Adaptive Control**: Προσαρμογή σε αλλαγές
        - **Reinforcement Learning**: Μάθηση optimal πολιτικής
        
        **5. Manipulation (Χειρισμός)**
        - **Grasping**: Πιάσιμο αντικειμένων
        - **Pick and Place**: Μεταφορά αντικειμένων
        - **Inverse Kinematics**: Υπολογισμός joint angles
        - **Force Control**: Έλεγχος δύναμης επαφής
        
        **6. Human-Robot Interaction (HRI)**
        - **Speech Recognition**: Φωνητικές εντολές
        - **Gesture Recognition**: Αναγνώριση χειρονομιών
        - **Emotion Detection**: Ανίχνευση συναισθημάτων
        - **Collaborative Robotics**: Cobots που δουλεύουν με ανθρώπους
        
        #### 🚗 Αυτόνομα Οχήματα (Autonomous Vehicles)
        
        **Επίπεδα Αυτονομίας:**
        - **Level 0**: Καμία αυτοματοποίηση
        - **Level 1**: Driver assistance (cruise control)
        - **Level 2**: Partial automation (Tesla Autopilot)
        - **Level 3**: Conditional automation
        - **Level 4**: High automation (συγκεκριμένες συνθήκες)
        - **Level 5**: Full automation (παντού)
        
        **Τεχνολογίες:**
        - **Sensors**: Κάμερες, LiDAR, Radar, Ultrasonic
        - **Perception**: Object detection, lane detection, traffic sign recognition
        - **Prediction**: Πρόβλεψη συμπεριφοράς άλλων οχημάτων/πεζών
        - **Planning**: Route planning, behavior planning
        - **Control**: Steering, throttle, brakes
        
        **Εταιρείες:**
        - Waymo (Google), Tesla, Cruise (GM), Argo AI, Zoox (Amazon)
        
        #### 🏭 Industrial Robotics (Βιομηχανικά Ρομπότ)
        
        **Τύποι:**
        - **Robotic Arms**: Manipulation, assembly
        - **AGVs** (Automated Guided Vehicles): Μεταφορά υλικών
        - **Collaborative Robots (Cobots)**: Εργασία με ανθρώπους
        - **Delta Robots**: Υψηλή ταχύτητα, pick-and-place
        
        **Εφαρμογές:**
        - Συναρμολόγηση (Automotive industry)
        - Welding, painting
        - Packaging
        - Quality inspection
        - Material handling
        
        **AI Enhancements:**
        - Computer Vision για inspection
        - Reinforcement Learning για βελτιστοποίηση
        - Predictive maintenance
        
        #### 🏠 Service Robotics (Ρομπότ Υπηρεσιών)
        
        **Household:**
        - **Vacuum Cleaners**: Roomba, Roborock
        - **Lawn Mowers**: Husqvarna, Worx
        - **Companion Robots**: Pepper, Jibo
        
        **Healthcare:**
        - **Surgical Robots**: Da Vinci Surgical System
        - **Rehabilitation Robots**: Βοήθεια σε ασθενείς
        - **Disinfection Robots**: UV-C για απολύμανση
        - **Delivery Robots**: Φάρμακα, γεύματα σε νοσοκομεία
        
        **Hospitality:**
        - Ρομπότ ρεσεψιόν σε ξενοδοχεία
        - Delivery robots σε εστιατόρια
        - Cleaning robots
        
        #### 🚁 Drones (UAVs - Unmanned Aerial Vehicles)
        
        **Εφαρμογές:**
        - **Photography/Videography**: Aerial shots
        - **Delivery**: Amazon Prime Air, Zipline (ιατρικά)
        - **Agriculture**: Crop monitoring, spraying
        - **Inspection**: Κτίρια, γέφυρες, πυλώνες
        - **Search and Rescue**: Εύρεση αγνοουμένων
        - **Military**: Surveillance, combat
        
        **AI Capabilities:**
        - Autonomous flight
        - Obstacle avoidance
        - Object tracking
        - Swarm intelligence (drone swarms)
        
        #### 🤝 Social Robots
        
        **Παραδείγματα:**
        - **Pepper**: Humanoid robot για interaction
        - **NAO**: Εκπαιδευτικό ρομπότ
        - **Sophia**: Από Hanson Robotics
        - **Paro**: Therapeutic seal robot
        
        **Capabilities:**
        - Facial recognition
        - Emotion detection
        - Natural language interaction
        - Educational content delivery
        
        #### 🧠 Key AI Techniques στη Ρομποτική
        
        **1. Reinforcement Learning**
        - Μάθηση πολιτικών από trial-and-error
        - Sim-to-real transfer
        - Παραδείγματα: Grasping, locomotion
        
        **2. Imitation Learning**
        - Μάθηση από demonstrations
        - Learning from human experts
        
        **3. Multi-Agent Systems**
        - Συντονισμός πολλαπλών ρομπότ
        - Swarm robotics
        
        **4. Sim-to-Real**
        - Training σε simulation
        - Transfer σε πραγματικό κόσμο
        - Domain randomization
        
        #### 🛠️ Platforms και Tools
        
        **Simulation:**
        - **Gazebo**: Robot simulation
        - **V-REP/CoppeliaSim**: Robot simulator
        - **PyBullet**: Physics simulation
        - **CARLA**: Autonomous driving simulator
        - **AirSim**: Drone and vehicle simulator
        
        **Frameworks:**
        - **ROS** (Robot Operating System): Middleware
        - **ROS 2**: Next-gen ROS
        - **OpenCV**: Computer vision
        - **PCL**: Point Cloud Library (3D data)
        
        **Hardware:**
        - **Raspberry Pi**: Low-cost computing
        - **NVIDIA Jetson**: Edge AI computing
        - **Arduino**: Microcontroller για actuators/sensors
        
        #### 🚧 Προκλήσεις
        
        - **Real-world Uncertainty**: Unpredictable περιβάλλοντα
        - **Safety**: Ασφάλεια ανθρώπων
        - **Generalization**: Λειτουργία σε διαφορετικά περιβάλλοντα
        - **Power Consumption**: Battery life
        - **Cost**: Ακριβά sensors και hardware
        - **Ethics**: Αυτονομία και ευθύνη
        
        #### 🔮 Μέλλον της Ρομποτικής
        
        - **General-purpose Robots**: Ρομπότ που κάνουν πολλές εργασίες
        - **Soft Robotics**: Ευέλικτα, ασφαλή υλικά
        - **Bio-inspired Robotics**: Μίμηση φύσης
        - **Nanorobots**: Ιατρικές εφαρμογές σε κυτταρικό επίπεδο
        - **Space Exploration**: Ρομπότ για exploration άλλων πλανητών
        """)
    
    st.markdown('---')
    
    st.markdown('---')
    section_title('1.2 Κύρια Δομικά Στοιχεία της Τεχνητής Νοημοσύνης')
    
    st.markdown("Κάντε κλικ σε κάθε στοιχείο για να μάθετε περισσότερα:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📊 Δεδομένα")
        concept_explainer(
            "Δεδομένα (Data)",
            "Τα δεδομένα είναι η **θεμελιώδης βάση** κάθε AI συστήματος. Χωρίς ποιοτικά δεδομένα, ακόμα και ο καλύτερος αλγόριθμος θα αποτύχει.",
            """
            **Τύποι Δεδομένων:**
            - **Structured Data**: Πίνακες, βάσεις δεδομένων (π.χ. CSV, SQL)
            - **Unstructured Data**: Κείμενο, εικόνες, βίντεο, ήχος
            - **Semi-structured Data**: JSON, XML
            
            **Ποιότητα Δεδομένων:**
            - **Accuracy**: Ακρίβεια και ορθότητα
            - **Completeness**: Πληρότητα (χωρίς missing values)
            - **Consistency**: Συνέπεια μεταξύ διαφορετικών πηγών
            - **Timeliness**: Επικαιρότητα
            - **Relevance**: Σχετικότητα με το πρόβλημα
            
            **Data Pipeline:**
            1. **Collection**: Συλλογή από πηγές
            2. **Cleaning**: Καθαρισμός (αφαίρεση duplicates, outliers)
            3. **Preprocessing**: Normalization, transformation
            4. **Augmentation**: Τεχνητή αύξηση dataset
            5. **Storage**: Αποθήκευση (Data Lakes, Warehouses)
            """,
            """
            - **Netflix**: 100+ million χρήστες, δισεκατομμύρια interactions
            - **Tesla**: Εκατομμύρια miles από αυτόνομη οδήγηση
            - **Google**: Τρισεκατομμύρια αναζητήσεις ετησίως
            - **Healthcare**: Ιατρικές εικόνες, patient records
            """
        )
    
    with col2:
        st.markdown("### ⚙️ Αλγόριθμοι")
        concept_explainer(
            "Αλγόριθμοι (Algorithms)",
            "Οι αλγόριθμοι είναι οι **μαθηματικές μέθοδοι** που μετατρέπουν τα δεδομένα σε χρήσιμες προβλέψεις και insights.",
            """
            **Κατηγορίες Αλγορίθμων:**
            
            **1. Supervised Learning:**
            - **Regression**: Linear, Polynomial, Ridge, Lasso
            - **Classification**: Logistic Regression, SVM, Decision Trees, Random Forest, XGBoost
            - **Neural Networks**: Feedforward, CNN, RNN
            
            **2. Unsupervised Learning:**
            - **Clustering**: K-Means, DBSCAN, Hierarchical
            - **Dimensionality Reduction**: PCA, t-SNE, UMAP
            - **Anomaly Detection**: Isolation Forest, One-Class SVM
            
            **3. Reinforcement Learning:**
            - **Q-Learning**: Value-based methods
            - **Policy Gradients**: REINFORCE, PPO, A3C
            - **Deep RL**: DQN, DDPG, SAC
            
            **Επιλογή Αλγορίθμου:**
            - Τύπος προβλήματος (classification, regression, clustering)
            - Μέγεθος dataset
            - Interpretability requirements
            - Computational resources
            - Real-time constraints
            """,
            """
            - **Linear Regression**: Πρόβλεψη τιμών ακινήτων
            - **Random Forest**: Credit scoring
            - **CNN**: Face recognition
            - **LSTM**: Stock price prediction
            - **Q-Learning**: Game playing AI
            """
        )
    
    with col3:
        st.markdown("### 🎯 Μοντέλα")
        concept_explainer(
            "Μοντέλα (Models)",
            "Τα μοντέλα είναι τα **εκπαιδευμένα συστήματα** που προκύπτουν από την εφαρμογή αλγορίθμων σε δεδομένα.",
            """
            **Lifecycle Μοντέλου:**
            
            **1. Training Phase:**
            - Feature engineering
            - Model selection
            - Hyperparameter tuning
            - Cross-validation
            - Training data split (train/validation/test)
            
            **2. Evaluation Phase:**
            - Performance metrics (Accuracy, Precision, Recall, F1, AUC)
            - Confusion matrix analysis
            - Learning curves
            - Error analysis
            
            **3. Deployment Phase:**
            - Model serialization (pickle, ONNX, TensorFlow SavedModel)
            - API creation (REST, gRPC)
            - Containerization (Docker)
            - Cloud deployment (AWS SageMaker, Azure ML, Google Vertex AI)
            
            **4. Monitoring Phase:**
            - Performance tracking
            - Data drift detection
            - Model retraining triggers
            - A/B testing
            
            **Model Types:**
            - **White-box**: Interpretable (Linear models, Decision Trees)
            - **Black-box**: High performance (Deep Neural Networks, Ensemble methods)
            - **Gray-box**: Balanced (XGBoost με SHAP values)
            """,
            """
            - **GPT-4**: 175B+ parameters, text generation
            - **YOLOv8**: Real-time object detection
            - **BERT**: Language understanding
            - **AlphaFold**: Protein structure prediction
            - **Stable Diffusion**: Image generation
            """
        )
    
    with col4:
        st.markdown("### 💻 Υποδομές")
        concept_explainer(
            "Υποδομές (Infrastructure)",
            "Η υποδομή περιλαμβάνει το **hardware και software** που απαιτούνται για training και deployment AI μοντέλων.",
            """
            **Hardware:**
            
            **1. CPUs (Central Processing Units)**:
            - General-purpose computing
            - Preprocessing, data loading
            - Inference για απλά μοντέλα
            
            **2. GPUs (Graphics Processing Units)**:
            - **NVIDIA**: A100, V100, RTX 4090
            - Parallel processing (1000s of cores)
            - Deep learning training
            - Speedup: 10-100x vs CPU
            
            **3. TPUs (Tensor Processing Units)**:
            - Google's custom AI chips
            - Optimized για matrix operations
            - Used in Google Cloud
            - Speedup: 15-30x vs GPUs
            
            **4. NPUs (Neural Processing Units)**:
            - Mobile devices (Apple Neural Engine, Qualcomm Hexagon)
            - Edge computing
            - Low power consumption
            
            **Software:**
            
            **Frameworks:**
            - TensorFlow, PyTorch, JAX
            - Keras, MXNet, PaddlePaddle
            
            **Cloud Platforms:**
            - **AWS**: SageMaker, EC2 P4d instances
            - **Google Cloud**: Vertex AI, TPU pods
            - **Azure**: Machine Learning, GPU VMs
            - **Specialized**: Lambda Labs, Paperspace, Vast.ai
            
            **MLOps Tools:**
            - **Experiment Tracking**: MLflow, Weights & Biases, Neptune
            - **Model Versioning**: DVC, Git LFS
            - **Deployment**: Docker, Kubernetes, KubeFlow
            - **Monitoring**: Prometheus, Grafana, Evidently AI
            
            **Storage:**
            - **S3, Google Cloud Storage**: Data lakes
            - **Databases**: PostgreSQL, MongoDB, Elasticsearch
            - **Feature Stores**: Feast, Tecton
            """,
            """
            **Κόστος Training:**
            - **GPT-3 Training**: ~$4.6M (estimated)
            - **Stable Diffusion**: ~$600K
            - **BERT-base**: ~$7K
            
            **Cloud Options:**
            - **Google Colab**: Free GPU/TPU για εκπαίδευση
            - **Kaggle Kernels**: Free GPU
            - **AWS Free Tier**: Limited credits
            """
        )
    
    st.markdown('---')
    
    st.markdown('---')
    section_title('1.3 Βασικά Ιστορικά Επιτεύγματα στο Χώρο της Τεχνητής Νοημοσύνης')
    
    timeline_data = {
        'Έτος': ['1950', '1956', '1997', '2011', '2012', '2016', '2018', '2020', '2022'],
        'Γεγονός': [
            'Turing Test',
            'Dartmouth Conference - Γέννηση της AI',
            'Deep Blue νικά τον Kasparov',
            'Watson νικά στο Jeopardy',
            'AlexNet - Deep Learning Revolution',
            'AlphaGo νικά τον Lee Sedol',
            'GPT-2 - Γλωσσικά μοντέλα',
            'GPT-3 - 175 billion παράμετροι',
            'ChatGPT - Μαζική υιοθέτηση AI'
        ]
    }
    
    df_timeline = pd.DataFrame(timeline_data)
    st.dataframe(df_timeline, use_container_width=True, hide_index=True)
    
    with st.expander('📖 Διαβάστε περισσότερα για την ιστορία'):
        st.markdown("""
        **1950 - Alan Turing**: Πρότεινε το Turing Test για να αξιολογήσει αν μια μηχανή μπορεί να σκέφτεται.
        
        **1956 - Dartmouth Conference**: Ο John McCarthy επινόησε τον όρο "Artificial Intelligence".
        
        **1997**: Το Deep Blue της IBM νίκησε τον παγκόσμιο πρωταθλητή σκακιού Garry Kasparov.
        
        **2012**: Η επανάσταση του Deep Learning με το AlexNet που νίκησε στο ImageNet competition.
        
        **2022-σήμερα**: Η εποχή των Large Language Models με ChatGPT να φέρνει την AI στην καθημερινότητα.
        """)
    
    st.markdown('---')
    section_title('1.4 Τεχνητή Νοημοσύνη: Εφαρμογές και Εξελίξεις')
    
    app_col1, app_col2 = st.columns(2)
    with app_col1:
        st.markdown("""
        #### 🏥 Υγεία
        - Διάγνωση ασθενειών από ιατρικές εικόνες
        - Ανακάλυψη νέων φαρμάκων
        - Προσωποποιημένη ιατρική
        - Πρόβλεψη επιδημιών
        
        #### 🚗 Μεταφορές
        - Αυτόνομα οχήματα
        - Βελτιστοποίηση διαδρομών
        - Έξυπνη διαχείριση κυκλοφορίας
        - Προβλεπτική συντήρηση
        """)
    with app_col2:
        st.markdown("""
        #### 💰 Χρηματοοικονομικά
        - Ανίχνευση απάτης
        - Αλγοριθμική διαπραγμάτευση
        - Πιστοδοτικός έλεγχος
        - Robo-advisors
        
        #### 🎓 Εκπαίδευση
        - Προσωποποιημένη μάθηση
        - Αυτόματη βαθμολόγηση
        - Εικονικοί βοηθοί διδασκαλίας
        - Προσβασιμότητα
        """)
    
    st.markdown('---')
    section_title('1.5 Βασικές Έννοιες — Πλαίσιο — Κανόνες')
    
    subsection_title('1.5.1 Βασικές Έννοιες')
    
    concepts_col1, concepts_col2, concepts_col3 = st.columns(3)
    with concepts_col1:
        st.markdown("""
        **Επιβλεπόμενη Μάθηση**
        (Supervised Learning)
        - Εκπαίδευση με labeled data
        - Πρόβλεψη outcomes
        - Παραδείγματα: Ταξινόμηση, Regression
        """)
    with concepts_col2:
        st.markdown("""
        **Μη Επιβλεπόμενη Μάθηση**
        (Unsupervised Learning)
        - Ανακάλυψη patterns
        - Clustering
        - Παραδείγματα: K-Means, PCA
        """)
    with concepts_col3:
        st.markdown("""
        **Ενισχυτική Μάθηση**
        (Reinforcement Learning)
        - Μάθηση μέσω δοκιμής-λάθους
        - Rewards/Penalties
        - Παραδείγματα: Gaming AI, Ρομποτική
        """)
    
    subsection_title('1.5.2 Πλαίσιο Εφαρμογής')
    st.markdown("""
    Η AI εφαρμόζεται σε διάφορα πλαίσια:
    - **Ερευνητικό**: Ανάπτυξη νέων αλγορίθμων και τεχνικών
    - **Βιομηχανικό**: Παραγωγικές εφαρμογές σε επιχειρήσεις
    - **Κοινωνικό**: Επίλυση κοινωνικών προβλημάτων
    - **Εκπαιδευτικό**: Μάθηση και εκπαίδευση
    """)
    
    subsection_title('1.5.3 Κανόνες και Ηθική')
    st.warning("""
    ⚠️ **Ηθικές Αρχές στην AI**:
    - Διαφάνεια (Transparency)
    - Δικαιοσύνη (Fairness)  
    - Ιδιωτικότητα (Privacy)
    - Ασφάλεια (Safety)
    - Λογοδοσία (Accountability)
    """)
    
    st.markdown('---')
    section_title('1.6 Πώς Λειτουργεί το ChatGPT')
    
    st.markdown("""
    ### 🤖 Το ChatGPT είναι ένα Large Language Model (LLM)
    
    **Βασικά Χαρακτηριστικά:**
    """)
    
    chatgpt_col1, chatgpt_col2 = st.columns(2)
    with chatgpt_col1:
        st.markdown("""
        **Αρχιτεκτονική:**
        - 🔄 Βασίζεται στην αρχιτεκτονική **Transformer**
        - 📊 Εκπαιδευμένο σε τεράστιο όγκο κειμένων
        - 🧮 Δισεκατομμύρια παράμετροι
        - 🎯 Fine-tuned με RLHF (Reinforcement Learning from Human Feedback)
        
        **Λειτουργία:**
        1. Λαμβάνει το input (prompt)
        2. Το μετατρέπει σε tokens
        3. Επεξεργάζεται μέσω transformer layers
        4. Προβλέπει το επόμενο token
        5. Παράγει συνεχή κείμενο
        """)
    with chatgpt_col2:
        st.markdown("""
        **Δυνατότητες:**
        - ✍️ Δημιουργία κειμένου
        - 💬 Συνομιλία
        - 📝 Σύνοψη
        - 🔄 Μετάφραση
        - 💻 Προγραμματισμός
        - 🎨 Δημιουργικότητα
        - 📊 Ανάλυση δεδομένων
        
        **Περιορισμοί:**
        - ⚠️ Μπορεί να παράγει λάθη
        - 📅 Cutoff date γνώσης
        - 🔍 Δεν αναζητά στο internet (GPT-3.5/4 base)
        - 🤔 Δεν κατανοεί πραγματικά
        """)
    
    with st.expander('🔬 Τεχνικές Λεπτομέρειες'):
        st.markdown("""
        **Pre-training:**
        - Εκπαίδευση σε τεράστια datasets
        - Unsupervised learning
        - Next token prediction
        
        **Fine-tuning:**
        - Supervised fine-tuning (SFT)
        - Reinforcement Learning from Human Feedback (RLHF)
        - Alignment με ανθρώπινες προτιμήσεις
        
        **Transformer Architecture:**
        ```
        Input → Tokenization → Embeddings → 
        Multi-Head Attention → Feed Forward → 
        Output Layer → Generated Text
        ```
        """)
    
    st.markdown('---')
    section_title('1.7 Βασικές Αρχές AI και Εφαρμογές')
    
    subsection_title('1.7.1 Machine Learning και Deep Learning')
    
    ml_col1, ml_col2 = st.columns(2)
    with ml_col1:
        st.markdown("""
        **Machine Learning (ML)**
        
        Είναι η ικανότητα των υπολογιστών να μαθαίνουν χωρίς να προγραμματίζονται ρητά.
        
        **Τύποι:**
        1. **Supervised Learning**
           - Classification
           - Regression
           
        2. **Unsupervised Learning**
           - Clustering
           - Dimensionality Reduction
           
        3. **Reinforcement Learning**
           - Agent-based learning
           - Rewards optimization
        """)
    with ml_col2:
        st.markdown("""
        **Deep Learning (DL)**
        
        Υποσύνολο του ML που χρησιμοποιεί νευρωνικά δίκτυα με πολλά επίπεδα (layers).
        
        **Αρχιτεκτονικές:**
        - **CNN** (Convolutional Neural Networks): Computer Vision
        - **RNN** (Recurrent Neural Networks): Sequences
        - **LSTM**: Long-term dependencies
        - **Transformers**: NLP, GPT, BERT
        - **GANs**: Generative models
        """)
    
    subsection_title('1.7.2 Δημιουργική Τεχνητή Νοημοσύνη (Generative AI)')
    
    st.markdown("""
    Η **Generative AI** αναφέρεται σε μοντέλα που μπορούν να δημιουργήσουν νέο περιεχόμενο.
    
    **Τύποι Generative AI:**
    """)
    
    gen_col1, gen_col2, gen_col3 = st.columns(3)
    with gen_col1:
        st.markdown("""
        **Text Generation**
        - GPT-4, ChatGPT
        - Claude
        - Gemini
        - LLaMA
        """)
    with gen_col2:
        st.markdown("""
        **Image Generation**
        - DALL-E
        - Midjourney
        - Stable Diffusion
        - Imagen
        """)
    with gen_col3:
        st.markdown("""
        **Other Modalities**
        - Music (MuseNet)
        - Video (Sora)
        - Voice (ElevenLabs)
        - Code (Copilot)
        """)
    
    subsection_title('1.7.3-1.7.7 Εφαρμογές AI σε Διάφορους Τομείς')
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['🏪 Πωλήσεις & Marketing', '📝 Γραμματεία', '💼 Επιχειρήσεις', '💵 Χρηματοοικονομικά', '⚕️ Υγεία'])
    
    with tab1:
        st.markdown("""
        **AI στις Πωλήσεις και Marketing**
        
        - **Personalization**: Εξατομικευμένες προτάσεις προϊόντων
        - **Chatbots**: Αυτόματη εξυπηρέτηση πελατών 24/7
        - **Predictive Analytics**: Πρόβλεψη συμπεριφοράς πελατών
        - **Content Generation**: Αυτόματη δημιουργία διαφημιστικού περιεχομένου
        - **Email Marketing**: Βελτιστοποίηση campaigns
        - **Social Media**: Ανάλυση sentiment, targeting
        """)
    
    with tab2:
        st.markdown("""
        **AI στη Γραμματεία και Διοικητικά Στελέχη**
        
        - **Έξυπνη Διαχείριση Εγγράφων**: Αυτόματη ταξινόμηση και αρχειοθέτηση
        - **Scheduling**: Έξυπνη διαχείριση ημερολογίου
        - **Transcription**: Μετατροπή ομιλίας σε κείμενο
        - **Translation**: Αυτόματη μετάφραση εγγράφων
        - **Data Entry**: Αυτοματοποίηση εισαγωγής δεδομένων
        """)
    
    with tab3:
        st.markdown("""
        **AI στη Ψηφιακή Μετασχηματισμό Επιχειρήσεων**
        
        - **Process Automation**: RPA (Robotic Process Automation)
        - **Decision Support**: Υποστήριξη λήψης αποφάσεων
        - **Supply Chain**: Βελτιστοποίηση εφοδιαστικής αλυσίδας
        - **Quality Control**: Αυτόματος έλεγχος ποιότητας
        - **Predictive Maintenance**: Προβλεπτική συντήρηση εξοπλισμού
        """)
    
    with tab4:
        st.markdown("""
        **AI στα Χρηματοοικονομικά**
        
        - **Fraud Detection**: Ανίχνευση απάτης σε πραγματικό χρόνο
        - **Algorithmic Trading**: Αυτοματοποιημένες συναλλαγές
        - **Credit Scoring**: Αξιολόγηση πιστοληπτικής ικανότητας
        - **Risk Management**: Διαχείριση κινδύνου
        - **Robo-Advisors**: Αυτόματη επενδυτική συμβουλευτική
        """)
    
    with tab5:
        st.markdown("""
        **AI στην Υγεία**
        
        - **Medical Imaging**: Διάγνωση από ακτινογραφίες, CT, MRI
        - **Drug Discovery**: Ανακάλυψη νέων φαρμάκων
        - **Personalized Medicine**: Εξατομικευμένη θεραπεία
        - **Patient Monitoring**: Παρακολούθηση ασθενών
        - **Clinical Decision Support**: Υποστήριξη ιατρικών αποφάσεων
        """)
    
    st.markdown('---')
    st.success("""
    🎓 **Συγχαρητήρια!** Ολοκληρώσατε το θεωρητικό μέρος. Συνεχίστε με τα πρακτικά παραδείγματα!
    """)

with tabs[1]:
    section_title('Παράδειγμα 1: Logistic Regression (Επιβλεπόμενη Μάθηση)')
    
    st.markdown("""
    Θα δημιουργήσουμε ένα μοντέλο **Logistic Regression** για **binary classification**.
    Αυτό είναι ένα κλασικό παράδειγμα επιβλεπόμενης μάθησης.
    """)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        n = st.slider('🔢 Αριθμός δειγμάτων', 100, 2000, 500, step=100)
        noise = st.slider('🔊 Επίπεδο θορύβου', 0.0, 0.5, 0.1, step=0.05)
    with col2:
        test_size = st.slider('📊 Ποσοστό Test Set', 0.1, 0.5, 0.3, step=0.05)
        random_state = st.slider('🎲 Random Seed', 1, 100, 42)
    
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, 
                                n_clusters_per_class=1, flip_y=noise, random_state=random_state)
    df = pd.DataFrame(X, columns=['Feature 1','Feature 2'])
    df['Target'] = y
    
    tab_viz1, tab_viz2, tab_viz3 = st.tabs(['📊 Δεδομένα', '📈 Οπτικοποίηση', '🎯 Μοντέλο'])
    
    with tab_viz1:
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown(f"**Στατιστικά:** {n} δείγματα, {df['Target'].value_counts().to_dict()}")
    
    with tab_viz2:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=0.6, edgecolors='k')
        ax.set_xlabel('Feature 1', fontsize=12)
        ax.set_ylabel('Feature 2', fontsize=12)
        ax.set_title('Scatter Plot των Δεδομένων', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Class')
        st.pyplot(fig)
        plt.close()
    
    with tab_viz3:
        if st.button('🚀 Εκπαίδευση Logistic Regression'):
            with st.spinner('Εκπαίδευση μοντέλου...'):
                time.sleep(1)  # Simulation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric('✅ Accuracy', f'{acc:.2%}')
                    st.metric('📦 Training Samples', len(X_train))
                with col_b:
                    st.metric('📊 Test Samples', len(X_test))
                    st.metric('🎯 Correct Predictions', int(acc * len(X_test)))
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_title('Confusion Matrix')
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)
                plt.close()
                
                # Visualization
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                scatter = ax2.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='coolwarm', 
                                     alpha=0.7, edgecolors='k', s=100)
                ax2.set_xlabel('Feature 1')
                ax2.set_ylabel('Feature 2')
                ax2.set_title('Προβλέψεις Μοντέλου')
                plt.colorbar(scatter, ax=ax2)
                st.pyplot(fig2)
                plt.close()
                
                st.success(f'✅ Το μοντέλο εκπαιδεύτηκε επιτυχώς με accuracy {acc:.2%}!')

    st.markdown('---')
    section_title('Παράδειγμα 2: K-Means Clustering (Μη Επιβλεπόμενη Μάθηση)')
    
    st.markdown("""
    Το **K-Means** είναι ένας δημοφιλής αλγόριθμος **clustering** (ομαδοποίησης) που ανήκει στη 
    μη επιβλεπόμενη μάθηση.
    """)
    
    col_km1, col_km2 = st.columns(2)
    with col_km1:
        n_samples_km = st.slider('📦 Αριθμός δειγμάτων', 100, 1000, 400, key='km_samples')
        n_clusters = st.slider('🎯 Αριθμός Clusters (K)', 2, 8, 3, key='kc')
    with col_km2:
        cluster_std = st.slider('📏 Cluster Standard Deviation', 0.5, 3.0, 1.5, key='km_std')
        random_km = st.slider('🎲 Random State', 1, 100, 42, key='km_random')
    
    if st.button('🔬 Εκτέλεση K-Means'):
        with st.spinner('Εκτέλεση αλγορίθμου...'):
            from sklearn.cluster import KMeans
            Xc, yc = make_blobs(n_samples=n_samples_km, centers=n_clusters, 
                               n_features=2, cluster_std=cluster_std, random_state=random_km)
            km = KMeans(n_clusters=n_clusters, random_state=random_km, n_init=10).fit(Xc)
            labels = km.labels_
            centers = km.cluster_centers_
            
            fig_km, ax_km = plt.subplots(figsize=(10, 6))
            scatter = ax_km.scatter(Xc[:,0], Xc[:,1], c=labels, cmap='tab10', alpha=0.6, edgecolors='k')
            ax_km.scatter(centers[:,0], centers[:,1], c='red', marker='X', s=300, 
                         edgecolors='black', linewidths=2, label='Centroids')
            ax_km.set_xlabel('Feature 1')
            ax_km.set_ylabel('Feature 2')
            ax_km.set_title(f'K-Means Clustering (K={n_clusters})')
            ax_km.legend()
            st.pyplot(fig_km)
            plt.close()
            
            st.success(f'✅ Τα δεδομένα ομαδοποιήθηκαν σε {n_clusters} clusters!')
            
            # Show cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            cluster_info = pd.DataFrame({'Cluster': unique, 'Size': counts})
            st.dataframe(cluster_info, use_container_width=True, hide_index=True)
    
    st.markdown('---')
    section_title('Παράδειγμα 3: Νευρωνικό Δίκτυο (Deep Learning)')
    
    st.markdown("""
    Ένα απλό **Neural Network** για classification. Θα χρησιμοποιήσουμε το scikit-learn 
    για να δημιουργήσουμε ένα multi-layer perceptron.
    """)
    
    col_nn1, col_nn2 = st.columns(2)
    with col_nn1:
        hidden_layers = st.multiselect('🧠 Hidden Layers (neurons)', 
                                       [10, 20, 50, 100, 200], default=[100, 50])
        activation = st.selectbox('⚡ Activation Function', ['relu', 'tanh', 'logistic'])
    with col_nn2:
        learning_rate = st.select_slider('📈 Learning Rate', 
                                         options=[0.0001, 0.001, 0.01, 0.1], value=0.001)
        max_iterations = st.slider('🔄 Max Iterations', 100, 1000, 200, step=100)
    
    if st.button('🚀 Train Neural Network'):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        
        with st.spinner('Training Neural Network...'):
            X_nn, y_nn = make_classification(n_samples=1000, n_features=20, n_informative=15,
                                            n_redundant=5, random_state=42)
            X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
                X_nn, y_nn, test_size=0.3, random_state=42)
            
            scaler = StandardScaler()
            X_train_nn = scaler.fit_transform(X_train_nn)
            X_test_nn = scaler.transform(X_test_nn)
            
            if hidden_layers:
                mlp = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers),
                                   activation=activation, 
                                   learning_rate_init=learning_rate,
                                   max_iter=max_iterations,
                                   random_state=42)
                mlp.fit(X_train_nn, y_train_nn)
                y_pred_nn = mlp.predict(X_test_nn)
                acc_nn = accuracy_score(y_test_nn, y_pred_nn)
                
                col_nn_a, col_nn_b, col_nn_c = st.columns(3)
                with col_nn_a:
                    st.metric('🎯 Test Accuracy', f'{acc_nn:.2%}')
                with col_nn_b:
                    st.metric('📊 Iterations', mlp.n_iter_)
                with col_nn_c:
                    st.metric('📉 Final Loss', f'{mlp.loss_:.4f}')
                
                # Plot loss curve
                fig_loss, ax_loss = plt.subplots(figsize=(10, 5))
                ax_loss.plot(mlp.loss_curve_, linewidth=2)
                ax_loss.set_xlabel('Iterations')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_title('Training Loss Curve')
                ax_loss.grid(True, alpha=0.3)
                st.pyplot(fig_loss)
                plt.close()
                
                st.success(f'✅ Neural Network trained with accuracy: {acc_nn:.2%}!')
            else:
                st.error('Παρακαλώ επιλέξτε τουλάχιστον ένα hidden layer!')

with tabs[2]:
    section_title('Εξομοιώσεις AI — Διαδραστικά Demos')
    
    st.markdown("""
    Σε αυτή την ενότητα μπορείτε να πειραματιστείτε με διάφορες παραμέτρους και να δείτε πώς 
    επηρεάζουν την απόδοση των μοντέλων AI.
    """)
    
    sim_option = st.selectbox('🎮 Επιλέξτε Εξομοίωση:', [
        'Επίδραση Θορύβου στην Ακρίβεια',
        'Overfitting vs Underfitting',
        'Επίδραση Μεγέθους Dataset',
        'Decision Boundary Visualization'
    ])
    
    if sim_option == 'Επίδραση Θορύβου στην Ακρίβεια':
        st.markdown('### 🔊 Πώς ο θόρυβος επηρεάζει το μοντέλο')
        
        noise_level = st.slider('Επίπεδο θορύβου (flip_y)', 0.0, 0.5, 0.1, 0.05)
        
        results = []
        for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            X_sim, y_sim = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                                               flip_y=noise, random_state=42)
            X_tr, X_te, y_tr, y_te = train_test_split(X_sim, y_sim, test_size=0.3, random_state=42)
            model_sim = LogisticRegression().fit(X_tr, y_tr)
            acc_sim = accuracy_score(y_te, model_sim.predict(X_te))
            results.append({'Noise': noise, 'Accuracy': acc_sim})
        
        df_results = pd.DataFrame(results)
        
        fig_noise, ax_noise = plt.subplots(figsize=(10, 6))
        ax_noise.plot(df_results['Noise'], df_results['Accuracy'], marker='o', linewidth=2, markersize=8)
        ax_noise.axvline(x=noise_level, color='r', linestyle='--', label=f'Current ({noise_level})')
        ax_noise.set_xlabel('Noise Level', fontsize=12)
        ax_noise.set_ylabel('Accuracy', fontsize=12)
        ax_noise.set_title('Επίδραση Θορύβου στην Ακρίβεια του Μοντέλου', fontsize=14, fontweight='bold')
        ax_noise.grid(True, alpha=0.3)
        ax_noise.legend()
        st.pyplot(fig_noise)
        plt.close()
        
        st.info(f'📊 Με noise level {noise_level}, η ακρίβεια είναι περίπου {df_results[df_results["Noise"]==noise_level]["Accuracy"].values[0]:.2%}')
    
    elif sim_option == 'Overfitting vs Underfitting':
        st.markdown('### ⚖️ Overfitting vs Underfitting')
        
        st.markdown("""
        - **Underfitting**: Το μοντέλο είναι πολύ απλό και δεν μαθαίνει καλά
        - **Good Fit**: Το μοντέλο είναι ισορροπημένο
        - **Overfitting**: Το μοντέλο έχει "μάθει" τον training set απ' έξω αλλά δεν γενικεύει
        """)
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        
        # Generate data
        np.random.seed(42)
        X_poly = np.sort(np.random.rand(50, 1) * 10, axis=0)
        y_poly = np.sin(X_poly).ravel() + np.random.randn(50) * 0.5
        
        degree = st.slider('🎯 Polynomial Degree', 1, 15, 3)
        
        poly_features = PolynomialFeatures(degree=degree)
        X_poly_transformed = poly_features.fit_transform(X_poly)
        model_poly = LinearRegression().fit(X_poly_transformed, y_poly)
        
        X_test_poly = np.linspace(0, 10, 100).reshape(-1, 1)
        X_test_poly_transformed = poly_features.transform(X_test_poly)
        y_pred_poly = model_poly.predict(X_test_poly_transformed)
        
        fig_poly, ax_poly = plt.subplots(figsize=(10, 6))
        ax_poly.scatter(X_poly, y_poly, color='blue', s=50, alpha=0.6, label='Data')
        ax_poly.plot(X_test_poly, y_pred_poly, color='red', linewidth=2, label=f'Degree {degree}')
        ax_poly.set_xlabel('X')
        ax_poly.set_ylabel('y')
        ax_poly.set_title(f'Polynomial Regression (Degree={degree})')
        ax_poly.legend()
        ax_poly.grid(True, alpha=0.3)
        st.pyplot(fig_poly)
        plt.close()
        
        mse = mean_squared_error(y_poly, model_poly.predict(X_poly_transformed))
        st.metric('📉 Training MSE', f'{mse:.4f}')
        
        if degree < 3:
            st.warning('⚠️ **Underfitting**: Το μοντέλο είναι πολύ απλό!')
        elif degree <= 5:
            st.success('✅ **Good Fit**: Το μοντέλο είναι ισορροπημένο!')
        else:
            st.error('🔴 **Overfitting**: Το μοντέλο είναι πολύ πολύπλοκο!')
    
    elif sim_option == 'Επίδραση Μεγέθους Dataset':
        st.markdown('### 📊 Πώς το μέγεθος του dataset επηρεάζει την απόδοση')
        
        sizes = [50, 100, 200, 500, 1000, 2000]
        accuracies = []
        
        with st.spinner('Εκτέλεση πειραμάτων...'):
            for size in sizes:
                X_size, y_size = make_classification(n_samples=size, n_features=10, random_state=42)
                X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_size, y_size, test_size=0.3, random_state=42)
                model_size = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr_s)
                acc_size = accuracy_score(y_te_s, model_size.predict(X_te_s))
                accuracies.append(acc_size)
        
        fig_size, ax_size = plt.subplots(figsize=(10, 6))
        ax_size.plot(sizes, accuracies, marker='o', linewidth=2, markersize=10, color='green')
        ax_size.set_xlabel('Dataset Size', fontsize=12)
        ax_size.set_ylabel('Accuracy', fontsize=12)
        ax_size.set_title('Επίδραση Μεγέθους Dataset στην Ακρίβεια', fontsize=14, fontweight='bold')
        ax_size.grid(True, alpha=0.3)
        st.pyplot(fig_size)
        plt.close()
        
        st.info('💡 **Παρατήρηση**: Με περισσότερα δεδομένα, το μοντέλο γενικά βελτιώνεται!')
    
    elif sim_option == 'Decision Boundary Visualization':
        st.markdown('### 🎯 Οπτικοποίηση Decision Boundary')
        
        from sklearn.svm import SVC
        
        kernel_choice = st.selectbox('Επιλέξτε Kernel:', ['linear', 'rbf', 'poly'])
        
        X_db, y_db = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                         n_clusters_per_class=1, random_state=42)
        
        model_db = SVC(kernel=kernel_choice, gamma='auto').fit(X_db, y_db)
        
        # Create mesh
        h = 0.02
        x_min, x_max = X_db[:, 0].min() - 1, X_db[:, 0].max() + 1
        y_min, y_max = X_db[:, 1].min() - 1, X_db[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = model_db.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig_db, ax_db = plt.subplots(figsize=(10, 8))
        ax_db.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        scatter = ax_db.scatter(X_db[:, 0], X_db[:, 1], c=y_db, cmap='coolwarm', 
                               edgecolors='k', s=50)
        ax_db.set_xlabel('Feature 1')
        ax_db.set_ylabel('Feature 2')
        ax_db.set_title(f'Decision Boundary (Kernel: {kernel_choice})')
        plt.colorbar(scatter, ax=ax_db)
        st.pyplot(fig_db)
        plt.close()
        
        acc_db = model_db.score(X_db, y_db)
        st.metric('🎯 Training Accuracy', f'{acc_db:.2%}')

with tabs[3]:
    section_title('Κουίζ Αυτοαξιολόγησης')
    
    st.markdown("""
    Δοκιμάστε τις γνώσεις σας! Απαντήστε στις παρακάτω ερωτήσεις για να ελέγξετε την κατανόησή σας.
    """)
    
    quiz_category = st.selectbox('📚 Επιλέξτε Κατηγορία:', [
        'Γενικά για AI',
        'Machine Learning',
        'ChatGPT & LLMs',
        'Εφαρμογές AI',
        'Ηθική & Κοινωνία'
    ])
    
    quizzes_general = [
        {
            'id':'q1', 'type':'mcq',
            'question':'Τι είναι η Μηχανική Μάθηση (Machine Learning);',
            'options':[
                'Γραφικά υπολογιστών',
                'Υποκατηγορία της AI που μαθαίνει από δεδομένα',
                'Μόνο δίκτυα νευρώνων',
                'Ένα είδος βάσης δεδομένων'
            ],
            'answer':'Υποκατηγορία της AI που μαθαίνει από δεδομένα',
            'explain':'Η ML είναι η υποκατηγορία της AI που επιτρέπει σε αλγορίθμους να μαθαίνουν από δεδομένα χωρίς να προγραμματίζονται ρητά.'
        },
        {
            'id':'q2', 'type':'tf',
            'question':'Η Τεχνητή Νοημοσύνη δημιουργήθηκε το 2020.',
            'answer':False,
            'explain':'Λάθος! Ο όρος "Artificial Intelligence" χρησιμοποιήθηκε για πρώτη φορά το 1956 στο Dartmouth Conference.'
        },
        {
            'id':'q3', 'type':'mcq',
            'question':'Ποιο από τα παρακάτω ΔΕΝ είναι βασικό δομικό στοιχείο της AI;',
            'options':[
                'Δεδομένα',
                'Αλγόριθμοι',
                'Μοντέλα',
                'Τυπογραφία'
            ],
            'answer':'Τυπογραφία',
            'explain':'Τα βασικά δομικά στοιχεία της AI είναι: Δεδομένα, Αλγόριθμοι, Μοντέλα και Υποδομές.'
        }
    ]
    
    quizzes_ml = [
        {
            'id':'q4', 'type':'mcq',
            'question':'Ποιος τύπος μάθησης χρησιμοποιεί labeled data;',
            'options':[
                'Unsupervised Learning',
                'Supervised Learning',
                'Reinforcement Learning',
                'Transfer Learning'
            ],
            'answer':'Supervised Learning',
            'explain':'Η Supervised Learning χρησιμοποιεί labeled data (δεδομένα με ετικέτες) για να εκπαιδεύσει μοντέλα.'
        },
        {
            'id':'q5', 'type':'tf',
            'question':'Το K-Means είναι αλγόριθμος επιβλεπόμενης μάθησης.',
            'answer':False,
            'explain':'Λάθος! Το K-Means είναι αλγόριθμος μη επιβλεπόμενης μάθησης (unsupervised) που χρησιμοποιείται για clustering.'
        },
        {
            'id':'q6', 'type':'mcq',
            'question':'Τι σημαίνει "Overfitting";',
            'options':[
                'Το μοντέλο είναι πολύ απλό',
                'Το μοντέλο έχει υψηλή ακρίβεια στο training set αλλά χαμηλή στο test set',
                'Το μοντέλο εκπαιδεύεται πολύ γρήγορα',
                'Το μοντέλο έχει πολλά features'
            ],
            'answer':'Το μοντέλο έχει υψηλή ακρίβεια στο training set αλλά χαμηλή στο test set',
            'explain':'Overfitting σημαίνει ότι το μοντέλο "έμαθε" τον training set απ\' έξω αλλά δεν μπορεί να γενικεύσει σε νέα δεδομένα.'
        }
    ]
    
    quizzes_chatgpt = [
        {
            'id':'q7', 'type':'tf',
            'question':'Το ChatGPT είναι transformer-based language model.',
            'answer':True,
            'explain':'Σωστά! Το ChatGPT βασίζεται στην αρχιτεκτονική Transformer που εισήχθη το 2017.'
        },
        {
            'id':'q8', 'type':'mcq',
            'question':'Τι σημαίνει "LLM";',
            'options':[
                'Large Language Model',
                'Low Level Machine',
                'Linear Learning Method',
                'Logical Language Mechanism'
            ],
            'answer':'Large Language Model',
            'explain':'LLM σημαίνει Large Language Model - μεγάλα γλωσσικά μοντέλα όπως το GPT-4, Claude, κλπ.'
        },
        {
            'id':'q9', 'type':'mcq',
            'question':'Τι είναι το "RLHF" που χρησιμοποιείται στο ChatGPT;',
            'options':[
                'Random Learning from Humans',
                'Reinforcement Learning from Human Feedback',
                'Rapid Language Handling Function',
                'Real-time Learning for High Frequency'
            ],
            'answer':'Reinforcement Learning from Human Feedback',
            'explain':'RLHF = Reinforcement Learning from Human Feedback. Χρησιμοποιείται για να "ευθυγραμμίσει" το μοντέλο με ανθρώπινες προτιμήσεις.'
        }
    ]
    
    quizzes_applications = [
        {
            'id':'q10', 'type':'mcq',
            'question':'Σε ποιον τομέα χρησιμοποιείται η AI για ανίχνευση απάτης;',
            'options':[
                'Εκπαίδευση',
                'Χρηματοοικονομικά',
                'Ψυχαγωγία',
                'Αθλητισμός'
            ],
            'answer':'Χρηματοοικονομικά',
            'explain':'Η AI χρησιμοποιείται ευρέως στα χρηματοοικονομικά για ανίχνευση απάτης σε πραγματικό χρόνο.'
        },
        {
            'id':'q11', 'type':'tf',
            'question':'Η AI μπορεί να διαγνώσει ασθένειες από ιατρικές εικόνες.',
            'answer':True,
            'explain':'Σωστά! Η AI (ειδικά Computer Vision) χρησιμοποιείται για να αναλύσει ακτινογραφίες, CT, MRI κλπ.'
        },
        {
            'id':'q12', 'type':'mcq',
            'question':'Ποια από τις παρακάτω ΔΕΝ είναι εφαρμογή Generative AI;',
            'options':[
                'DALL-E (δημιουργία εικόνων)',
                'ChatGPT (δημιουργία κειμένου)',
                'Spam Filter (φιλτράρισμα email)',
                'Midjourney (δημιουργία εικόνων)'
            ],
            'answer':'Spam Filter (φιλτράρισμα email)',
            'explain':'Το spam filter είναι classification task, όχι generative. Τα άλλα τρία δημιουργούν νέο περιεχόμενο.'
        }
    ]
    
    quizzes_ethics = [
        {
            'id':'q13', 'type':'tf',
            'question':'Η AI μπορεί να είναι μεροληπτική (biased) αν εκπαιδευτεί σε μεροληπτικά δεδομένα.',
            'answer':True,
            'explain':'Σωστά! Το bias στα δεδομένα μεταφέρεται στο μοντέλο. Γι\' αυτό είναι σημαντική η δικαιοσύνη και διαφάνεια στην AI.'
        },
        {
            'id':'q14', 'type':'mcq',
            'question':'Ποια ηθική αρχή αφορά το δικαίωμα των ανθρώπων να γνωρίζουν πώς παίρνονται αποφάσεις από AI;',
            'options':[
                'Privacy',
                'Transparency',
                'Accountability',
                'Fairness'
            ],
            'answer':'Transparency',
            'explain':'Η Transparency (Διαφάνεια) σημαίνει ότι τα AI συστήματα πρέπει να είναι κατανοητά και explainable.'
        },
        {
            'id':'q15', 'type':'mcq',
            'question':'Ποιο είναι το μεγαλύτερο ηθικό πρόβλημα με τα Large Language Models;',
            'options':[
                'Κατανάλωση ενέργειας',
                'Παραγωγή misinformation και hallucinations',
                'Κόστος training',
                'Ταχύτητα απόκρισης'
            ],
            'answer':'Παραγωγή misinformation και hallucinations',
            'explain':'Τα LLMs μπορεί να παράγουν πληροφορίες που ακούγονται πειστικές αλλά είναι λανθασμένες (hallucinations).'
        }
    ]
    
    if quiz_category == 'Γενικά για AI':
        quizzes_to_show = quizzes_general
    elif quiz_category == 'Machine Learning':
        quizzes_to_show = quizzes_ml
    elif quiz_category == 'ChatGPT & LLMs':
        quizzes_to_show = quizzes_chatgpt
    elif quiz_category == 'Εφαρμογές AI':
        quizzes_to_show = quizzes_applications
    else:
        quizzes_to_show = quizzes_ethics
    
    for i, q in enumerate(quizzes_to_show, 1):
        with st.container():
            st.markdown(f"#### Ερώτηση {i}")
            show_quiz(q)
            st.markdown('---')

with tabs[4]:
    section_title('Διαδραστικές Ασκήσεις')
    
    st.markdown("""
    Εδώ μπορείτε να πειραματιστείτε με πραγματικά προβλήματα και να εφαρμόσετε τις γνώσεις σας!
    
    💡 **Tip**: Κάντε κλικ στο "Open in Colab" για να δοκιμάσετε hands-on εκπαίδευση με πραγματικό κώδικα!
    """)
    
    # Colab Notebooks Section
    st.markdown('---')
    st.markdown('### 📓 Google Colab Notebooks - Hands-On Εκπαίδευση')
    
    st.info("""
    🎓 **Τα Google Colab notebooks προσφέρουν:**
    - Δωρεάν GPU/TPU για εκπαίδευση μοντέλων
    - Προ-εγκατεστημένες βιβλιοθήκες (TensorFlow, PyTorch, scikit-learn)
    - Βήμα-προς-βήμα οδηγίες με εξηγήσεις
    - Έτοιμο κώδικα που μπορείτε να τροποποιήσετε
    - Αποθήκευση στο Google Drive σας
    """)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("#### 🚀 Beginner Level")
        
        colab_button(
            "Linear Regression - Βασικά",
            "https://colab.research.google.com/github/ageron/handson-ml2/blob/master/04_training_linear_models.ipynb",
            "Μάθετε Linear Regression από το μηδέν με scikit-learn"
        )
        
        st.markdown("---")
        
        colab_button(
            "K-Means Clustering",
            "https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb",
            "Unsupervised learning με K-Means algorithm"
        )
        
        st.markdown("---")
        
        colab_button(
            "Decision Trees & Random Forests",
            "https://colab.research.google.com/github/ageron/handson-ml2/blob/master/06_decision_trees.ipynb",
            "Ensemble methods για classification και regression"
        )
    
    with col_c2:
        st.markdown("#### 🔥 Advanced Level")
        
        colab_button(
            "Neural Networks με TensorFlow",
            "https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb",
            "Δημιουργήστε το πρώτο σας Deep Learning μοντέλο"
        )
        
        st.markdown("---")
        
        colab_button(
            "CNN για Image Classification",
            "https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/cnn.ipynb",
            "Convolutional Neural Networks για ταξινόμηση εικόνων"
        )
        
        st.markdown("---")
        
        colab_button(
            "NLP με Transformers",
            "https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification.ipynb",
            "Sentiment analysis με pre-trained BERT"
        )
    
    st.markdown('---')
    st.markdown('### 🎮 Διαδραστικές Εξασκήσεις (In-App)')
    
    exercise_choice = st.selectbox('🎯 Επιλέξτε Άσκηση:', [
        'Πρόβλεψη Τιμών (Regression)',
        'Ταξινόμηση Εικόνων (Image Classification Simulation)',
        'Sentiment Analysis Simulator',
        'Δημιουργία Συστήματος Συστάσεων',
        '🆕 Custom ML Pipeline Builder'
    ])
    
    if exercise_choice == 'Πρόβλεψη Τιμών (Regression)':
        st.markdown('### 🏠 Πρόβλεψη Τιμών Ακινήτων')
        
        st.markdown("""
        Σε αυτή την άσκηση θα δημιουργήσετε ένα μοντέλο που προβλέπει την τιμή ενός ακινήτου 
        βάσει των χαρακτηριστικών του.
        """)
        
        # Add interactive concept explainer
        concept_explainer(
            "Linear Regression",
            "Η **Linear Regression** είναι ένας αλγόριθμος supervised learning που προβλέπει μια συνεχή τιμή (continuous value).",
            """
            **Μαθηματική Φόρμουλα:**
            ```
            y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
            ```
            
            Όπου:
            - y: Target variable (τιμή)
            - x: Features (μέγεθος, δωμάτια, κλπ.)
            - β: Coefficients (παράμετροι που μαθαίνει το μοντέλο)
            - ε: Error term
            
            **Πώς λειτουργεί:**
            1. Το μοντέλο προσπαθεί να βρει την "καλύτερη γραμμή" που ταιριάζει στα δεδομένα
            2. "Καλύτερη" σημαίνει ελαχιστοποίηση του Mean Squared Error (MSE)
            3. Χρησιμοποιεί Gradient Descent ή Normal Equation
            
            **Πλεονεκτήματα:**
            - Απλό και γρήγορο
            - Interpretable (μπορούμε να δούμε τις παραμέτρους)
            - Καλό baseline μοντέλο
            
            **Μειονεκτήματα:**
            - Υποθέτει γραμμική σχέση
            - Sensitive σε outliers
            - Δεν μπορεί να πιάσει πολύπλοκα patterns
            """,
            """
            - Πρόβλεψη τιμών ακινήτων
            - Sales forecasting
            - Stock price trends (short-term)
            - Energy consumption prediction
            """
        )
        
        # Simulate housing data
        np.random.seed(42)
        n_houses = st.slider('Πόσα ακίνητα στο dataset;', 50, 500, 200)
        
        size = np.random.randint(50, 300, n_houses)
        rooms = np.random.randint(1, 6, n_houses)
        age = np.random.randint(0, 50, n_houses)
        
        price = (size * 1000 + rooms * 15000 - age * 500 + 
                 np.random.randn(n_houses) * 10000)
        
        df_houses = pd.DataFrame({
            'Μέγεθος (τ.μ.)': size,
            'Δωμάτια': rooms,
            'Ηλικία (έτη)': age,
            'Τιμή (€)': price
        })
        
        st.dataframe(df_houses.head(10))
        
        if st.button('🚀 Εκπαίδευση Μοντέλου Πρόβλεψης'):
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, r2_score
            
            X_houses = df_houses[['Μέγεθος (τ.μ.)', 'Δωμάτια', 'Ηλικία (έτη)']].values
            y_houses = df_houses['Τιμή (€)'].values
            
            X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
                X_houses, y_houses, test_size=0.3, random_state=42)
            
            model_h = LinearRegression().fit(X_train_h, y_train_h)
            y_pred_h = model_h.predict(X_test_h)
            
            mae = mean_absolute_error(y_test_h, y_pred_h)
            r2 = r2_score(y_test_h, y_pred_h)
            
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.metric('📊 R² Score', f'{r2:.3f}')
                concept_explainer(
                    "R² Score (Coefficient of Determination)",
                    "Το R² μετρά πόσο καλά το μοντέλο εξηγεί τη διακύμανση των δεδομένων.",
                    """
                    **Τιμές R²:**
                    - **1.0**: Τέλεια πρόβλεψη
                    - **0.8-0.9**: Πολύ καλό μοντέλο
                    - **0.6-0.8**: Καλό μοντέλο
                    - **< 0.5**: Αδύναμο μοντέλο
                    - **< 0**: Χειρότερο από απλό mean
                    
                    **Φόρμουλα:**
                    ```
                    R² = 1 - (Σ(y_actual - y_pred)²) / (Σ(y_actual - y_mean)²)
                    ```
                    """,
                    f"Το δικό σας μοντέλο με R²={r2:.3f} θεωρείται {'εξαιρετικό!' if r2>0.9 else 'πολύ καλό!' if r2>0.7 else 'καλό' if r2>0.5 else 'αρκετό για βελτίωση'}"
                )
            with col_h2:
                st.metric('💰 Mean Absolute Error', f'{mae:,.0f} €')
                concept_explainer(
                    "Mean Absolute Error (MAE)",
                    "Το MAE είναι ο μέσος απόλυτος λάθος των προβλέψεων σε πραγματικές μονάδες.",
                    """
                    **Τι σημαίνει:**
                    - Κατά μέσο όρο, οι προβλέψεις μας απέχουν MAE € από την πραγματική τιμή
                    - Πιο εύκολο να ερμηνευτεί από MSE (Mean Squared Error)
                    - Δεν τιμωρεί τα μεγάλα λάθη τόσο όσο το MSE
                    
                    **Φόρμουλα:**
                    ```
                    MAE = (1/n) * Σ|y_actual - y_pred|
                    ```
                    
                    **Πότε είναι καλό:**
                    - Όσο μικρότερο τόσο καλύτερα
                    - Συγκρίνετέ το με το range των τιμών
                    - Αν MAE << std(y), είναι πολύ καλό
                    """,
                    f"Με μέση τιμή {y_houses.mean():,.0f}€ και MAE {mae:,.0f}€, το σφάλμα είναι {(mae/y_houses.mean()*100):.1f}% της μέσης τιμής"
                )
            
            fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
            ax_pred.scatter(y_test_h, y_pred_h, alpha=0.6, edgecolors='k')
            ax_pred.plot([y_test_h.min(), y_test_h.max()], 
                        [y_test_h.min(), y_test_h.max()], 
                        'r--', lw=2, label='Perfect Prediction')
            ax_pred.set_xlabel('Πραγματική Τιμή (€)')
            ax_pred.set_ylabel('Προβλεπόμενη Τιμή (€)')
            ax_pred.set_title('Πραγματικές vs Προβλεπόμενες Τιμές')
            ax_pred.legend()
            ax_pred.grid(True, alpha=0.3)
            st.pyplot(fig_pred)
            plt.close()
            
            st.success('✅ Μοντέλο εκπαιδεύτηκε επιτυχώς!')
            
            # Interactive prediction
            st.markdown('### 🏡 Δοκιμάστε μια Πρόβλεψη')
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                new_size = st.number_input('Μέγεθος (τ.μ.)', 50, 300, 100)
            with col_p2:
                new_rooms = st.number_input('Δωμάτια', 1, 5, 3)
            with col_p3:
                new_age = st.number_input('Ηλικία (έτη)', 0, 50, 10)
            
            if st.button('🔮 Πρόβλεψη Τιμής'):
                new_pred = model_h.predict([[new_size, new_rooms, new_age]])[0]
                st.balloons()
                st.success(f'🏠 Εκτιμώμενη Τιμή: **{new_pred:,.0f} €**')
    
    elif exercise_choice == 'Ταξινόμηση Εικόνων (Image Classification Simulation)':
        st.markdown('### 📸 Simulation: Ταξινόμηση Εικόνων')
        
        st.markdown("""
        Προσομοίωση ενός image classifier. Στην πράξη θα χρησιμοποιούσαμε CNN (Convolutional Neural Networks).
        """)
        
        st.info("""
        **Παράδειγμα Workflow:**
        1. Load dataset (π.χ. MNIST, CIFAR-10)
        2. Preprocess εικόνες (normalization, augmentation)
        3. Δημιουργία CNN architecture
        4. Training με backpropagation
        5. Evaluation και fine-tuning
        """)
        
        # Simulate image classification
        categories = ['Γάτα', 'Σκύλος', 'Πουλί', 'Ψάρι', 'Αυτοκίνητο']
        
        num_images = st.slider('Αριθμός εικόνων στο dataset', 100, 1000, 500)
        
        # Simulate accuracy over epochs
        epochs = 20
        train_acc = []
        val_acc = []
        
        for epoch in range(epochs):
            train_acc.append(min(0.5 + epoch * 0.025 + np.random.rand() * 0.02, 0.98))
            val_acc.append(min(0.4 + epoch * 0.022 + np.random.rand() * 0.03, 0.92))
        
        fig_training, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(range(1, epochs+1), train_acc, label='Training Accuracy', marker='o')
        ax1.plot(range(1, epochs+1), val_acc, label='Validation Accuracy', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confusion matrix simulation
        cm_sim = np.random.randint(0, 50, (len(categories), len(categories)))
        np.fill_diagonal(cm_sim, np.random.randint(80, 120, len(categories)))
        
        sns.heatmap(cm_sim, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=categories, yticklabels=categories, ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        st.pyplot(fig_training)
        plt.close()
        
        st.success(f'✅ Final Validation Accuracy: {val_acc[-1]:.2%}')
    
    elif exercise_choice == 'Sentiment Analysis Simulator':
        st.markdown('### 💬 Ανάλυση Συναισθήματος (Sentiment Analysis)')
        
        st.markdown("""
        Εισάγετε ένα κείμενο και το "μοντέλο" θα προσπαθήσει να εντοπίσει το συναίσθημα!
        """)
        
        user_text = st.text_area('✍️ Γράψτε ένα κείμενο:', 
                                 'Αυτή η εφαρμογή AI είναι φανταστική! Μαθαίνω πολλά!')
        
        if st.button('🔍 Ανάλυση Συναισθήματος'):
            # Simple rule-based sentiment (simulation)
            positive_words = ['καλό', 'φανταστικό', 'εξαιρετικό', 'υπέροχο', 'άριστο', 
                            'χαρούμενος', 'ικανοποιημένος', 'μαθαίνω', 'ευχαριστώ']
            negative_words = ['κακό', 'άσχημο', 'απαίσιο', 'δύσκολο', 'πρόβλημα',
                            'λυπημένος', 'απογοητευμένος', 'λάθος']
            
            text_lower = user_text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                sentiment = '😊 Θετικό'
                color = 'green'
                score = min((pos_count / (pos_count + neg_count + 1)) * 100, 95)
            elif neg_count > pos_count:
                sentiment = '😞 Αρνητικό'
                color = 'red'
                score = min((neg_count / (pos_count + neg_count + 1)) * 100, 95)
            else:
                sentiment = '😐 Ουδέτερο'
                color = 'gray'
                score = 50
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f'### Συναίσθημα: {sentiment}')
            with col_s2:
                st.metric('Βαθμολογία Βεβαιότητας', f'{score:.0f}%')
            
            st.progress(score / 100)
            
            st.info("""
            **Σημείωση**: Αυτό είναι ένα απλοποιημένο παράδειγμα. Πραγματικά sentiment analysis 
            μοντέλα χρησιμοποιούν:
            - Deep Learning (LSTM, Transformers)
            - Pre-trained models (BERT, RoBERTa)
            - Context understanding
            - Multilingual support
            """)
    
    elif exercise_choice == 'Δημιουργία Συστήματος Συστάσεων':
        st.markdown('### 🎬 Σύστημα Συστάσεων (Recommendation System)')
        
        st.markdown("""
        Προσομοίωση ενός recommendation system για ταινίες.
        """)
        
        # Sample movies
        movies = {
            'Action': ['Mad Max', 'John Wick', 'Die Hard', 'Terminator'],
            'Drama': ['The Godfather', 'Schindler\'s List', 'Forrest Gump'],
            'Comedy': ['The Hangover', 'Superbad', 'Bridesmaids'],
            'Sci-Fi': ['Inception', 'Interstellar', 'The Matrix', 'Blade Runner']
        }
        
        st.markdown('#### Βήμα 1: Επιλέξτε ταινίες που σας αρέσουν')
        
        liked_movies = []
        for genre, movie_list in movies.items():
            selected = st.multiselect(f'{genre}:', movie_list, key=genre)
            liked_movies.extend([(movie, genre) for movie in selected])
        
        if st.button('🎯 Λήψη Συστάσεων') and liked_movies:
            st.markdown('#### 🎬 Προτεινόμενες Ταινίες:')
            
            # Count genres
            from collections import Counter
            genres_count = Counter([genre for _, genre in liked_movies])
            top_genre = genres_count.most_common(1)[0][0] if genres_count else 'Action'
            
            # Recommend from top genre
            recommendations = [m for m in movies[top_genre] 
                             if m not in [movie for movie, _ in liked_movies]][:3]
            
            for rec in recommendations:
                st.success(f'✨ {rec} ({top_genre})')
            
            st.info(f'💡 Βασισμένο στις προτιμήσεις σας για {top_genre}!')
            
            # Visualize preferences
            fig_pref, ax_pref = plt.subplots(figsize=(8, 5))
            genres = list(genres_count.keys())
            counts = list(genres_count.values())
            ax_pref.bar(genres, counts, color='skyblue', edgecolor='black')
            ax_pref.set_xlabel('Genre')
            ax_pref.set_ylabel('Number of Liked Movies')
            ax_pref.set_title('Your Genre Preferences')
            st.pyplot(fig_pref)
            plt.close()

with tabs[5]:
    # Import ΕΜΠΛΟΥΤΙΣΜΕΝΟ chatbot module με ΠΛΗΡΗ γνώση
    try:
        from chatbot_simple import create_chatbot_interface as create_enriched_chatbot_interface
        
        section_title('🌟 AI Knowledge Assistant - Εμπλουτισμένη Έκδοση')
        
        st.markdown("""
        ### Καλώς ήρθατε στον **Εμπλουτισμένο AI Knowledge Assistant**! 🤖✨
        
        Αυτός ο προηγμένος intelligent chatbot έχει:
        - 📚 **Πρόσβαση στο ΠΛΗΡΕΣ εκπαιδευτικό υλικό** (957 σελίδες PDF)
        - 🌐 **Internet access** σε Wikipedia, ArXiv, και curated AI resources
        - 🧠 **Βαθιά κατανόηση** όλων των ενοτήτων (1.1-1.7)
        - 💬 **Διαδραστική συνομιλία** με context awareness
        
        ### 🎯 Νέες Δυνατότητες:
    
        
        - ✅ **Ολοκληρωμένες απαντήσεις** με παραδείγματα και use cases
        - ✅ **Ιστορική προοπτική** της AI
        - ✅ **Σύγκριση τεχνολογιών** (CNN vs RNN vs Transformers)
        - ✅ **Ηθικά ζητήματα** και προκλήσεις
        - ✅ **Πρακτικοί πόροι** (documentation, courses, papers)
        
        ### 📚 Θέματα που καλύπτει:
        
        **Θεωρία:**
        - Ορισμός και τύποι AI (Narrow, General, Super)
        - Βασικά δομικά στοιχεία (Δεδομένα, Αλγόριθμοι, Μοντέλα, Computing)
        - Machine Learning (Supervised, Unsupervised, Reinforcement)
        - Deep Learning (CNN, RNN, LSTM, Transformers)
        - ChatGPT και Large Language Models
        - Generative AI (GANs, VAEs, Diffusion Models)
        
        **Πράξη:**
        - Εφαρμογές σε Υγεία, Εκπαίδευση, Finance, Marketing
        - Python & ML frameworks (TensorFlow, PyTorch)
        - Google Colab notebooks
        - Πρακτικά παραδείγματα
        
        **Ηθική:**
        - Bias & Fairness
        - Privacy & Security (GDPR)
        - Transparency & Explainability
        - Job displacement & Future of work
        
        ---
        """)
        
        # Create ENRICHED chatbot interface
        create_enriched_chatbot_interface()
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            ✅ **Πλεονεκτήματα:**
            - Πρόσβαση σε 957 σελίδες εκπαιδευτικού υλικού
            - Online resources από αξιόπιστες πηγές
            - Διαδραστική μάθηση
            - Άμεσες απαντήσεις
            """)
        with col2:
            st.info("""
            💡 **Tip**: 
            Κάντε συγκεκριμένες ερωτήσεις για καλύτερα αποτελέσματα!
            
            Π.χ. "Εξήγησε την αρχιτεκτονική Transformer" 
            αντί για "Πες μου για AI"
            """)
        
    except (ImportError, Exception) as e:
        # Fallback to old chatbot
        from chatbot import create_chatbot_interface
        
        section_title('AI Knowledge Assistant - Ρωτήστε με οτιδήποτε!')
        
        st.warning(f"⚠️ Το εμπλουτισμένο chatbot δεν είναι διαθέσιμο. Χρήση βασικής έκδοσης...")
        
        st.markdown("""
        Καλώς ήρθατε στον **AI Knowledge Assistant**! 🤖
        
        Αυτός ο intelligent chatbot έχει πρόσβαση στο πλήρες εκπαιδευτικό υλικό και μπορεί να απαντήσει
        σε ερωτήσεις σχετικά με την Τεχνητή Νοημοσύνη.
        
        ### 💡 Τι μπορεί να κάνει:
    
        - ✅ Απαντά σε ερωτήσεις για AI concepts
        - ✅ Εξηγεί τεχνικούς όρους με παραδείγματα
        - ✅ Παρέχει εις βάθος αναλύσεις
        - ✅ Συνδέει διάφορες έννοιες μεταξύ τους
        
        ### 🎯 Θέματα που καλύπτει:
        
        - Βασικά δομικά στοιχεία της AI
        - Machine Learning (Supervised, Unsupervised, Reinforcement)
        - Deep Learning και αρχιτεκτονικές
        - ChatGPT και Large Language Models
        - Πρακτικές εφαρμογές σε διάφορους τομείς
        - Αλγόριθμοι και τεχνικές
        
        ---
        """)
        
        # Create chatbot interface
        create_chatbot_interface()
        
        st.markdown("---")
        st.info("""
        💡 **Tip**: Το chatbot χρησιμοποιεί το εκπαιδευτικό υλικό από το PDF για να παρέχει ακριβείς απαντήσεις.
        Για πιο προηγμένες ερωτήσεις, δείτε τις ενότητες "Περιεχόμενο" και "Concept Explainers".
        """)

with tabs[6]:
    section_title('Πόροι & Οδηγίες')
    
    st.markdown("""
    ## 📚 Πρόσθετοι Πόροι για Μάθηση
    
    ### 🌐 Online Courses
    - **Coursera**: Machine Learning by Andrew Ng
    - **Fast.ai**: Practical Deep Learning
    - **DeepLearning.AI**: Deep Learning Specialization
    - **Udacity**: AI Programming with Python
    
    ### 📖 Βιβλία
    - "Hands-On Machine Learning" - Aurélien Géron
    - "Deep Learning" - Ian Goodfellow
    - "Pattern Recognition and Machine Learning" - Christopher Bishop
    - "AI: A Modern Approach" - Russell & Norvig
    
    ### 💻 Frameworks & Tools
    - **TensorFlow**: Deep learning framework by Google
    - **PyTorch**: Deep learning framework by Meta
    - **Scikit-learn**: Machine learning library
    - **Keras**: High-level neural networks API
    - **Hugging Face**: Pre-trained models
    
    ### 🎓 Πανεπιστημιακά Μαθήματα (Free)
    - MIT OpenCourseWare: Introduction to AI
    - Stanford CS229: Machine Learning
    - Berkeley CS188: Introduction to AI
    
    ### 🤝 Communities
    - **Kaggle**: Competitions & Datasets
    - **GitHub**: Open source projects
    - **Stack Overflow**: Q&A
    - **Reddit**: r/MachineLearning, r/artificial
    
    ### 📊 Datasets
    - **UCI Machine Learning Repository**
    - **Kaggle Datasets**
    - **Google Dataset Search**
    - **ImageNet**
    - **COCO Dataset**
    
    ---
    
    ## 🚀 Οδηγίες Χρήσης Εφαρμογής
    
    1. **Περιεχόμενο**: Διαβάστε τη θεωρία για τις ενότητες 1.1-1.7
    2. **Παραδείγματα Python**: Πειραματιστείτε με πραγματικά παραδείγματα ML
    3. **Εξομοιώσεις**: Δείτε πώς οι παράμετροι επηρεάζουν τα μοντέλα
    4. **Κουίζ**: Ελέγξτε τις γνώσεις σας
    5. **Διαδραστικές Ασκήσεις**: Εφαρμόστε τις γνώσεις σας σε πρακτικά προβλήματα
    
    ---
    
    ## 📝 Για Offline Χρήση
    
    Αυτή η εφαρμογή μπορεί να τρέξει εντελώς offline. Για να την εγκαταστήσετε:
    
    ```bash
    # Εγκατάσταση dependencies
    pip install streamlit scikit-learn matplotlib numpy pandas seaborn
    
    # Εκτέλεση εφαρμογής
    streamlit run ai_training_app.py
    ```
    
    ---
    
    ## 🎯 Επόμενα Βήματα
    
    1. Ολοκληρώστε όλα τα κουίζ
    2. Δοκιμάστε όλες τις εξομοιώσεις
    3. Κάντε τις διαδραστικές ασκήσεις
    4. Εξερευνήστε τους προτεινόμενους πόρους
    5. Ξεκινήστε το δικό σας AI project!
    
    ---
    
    ## 📧 Επικοινωνία & Υποστήριξη
    
    Για ερωτήσεις και υποστήριξη, ανατρέξτε στο εκπαιδευτικό υλικό και τους πόρους που παρέχονται.
    
    **Καλή επιτυχία στην εκμάθηση AI!** 🚀🤖
    """)
    
    # Footer
    st.markdown('---')
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🤖 AI Training Application v2.0</p>
        <p>Δημιουργήθηκε με ❤️ χρησιμοποιώντας Streamlit</p>
        <p>Βασισμένο στο εκπαιδευτικό υλικό: "Εφαρμογές Τεχνητής Νοημοσύνης και ChatGPT σε Κρίσιμους Τομείς"</p>
    </div>
    """, unsafe_allow_html=True)
