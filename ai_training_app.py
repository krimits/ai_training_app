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

tabs = st.tabs(['📚 Περιεχόμενο','🐍 Παραδείγματα Python','🔬 Εξομοιώσεις AI','✅ Κουίζ','💡 Διαδραστικές Ασκήσεις','📖 Πόροι'])

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
        - 🧠 **Machine Learning (ML)**: Μηχανές που μαθαίνουν από δεδομένα
        - 🌐 **Deep Learning**: Νευρωνικά δίκτυα με πολλά επίπεδα  
        - 💬 **Natural Language Processing (NLP)**: Επεξεργασία φυσικής γλώσσας
        - 👁️ **Computer Vision**: Όραση υπολογιστών
        - 🤖 **Robotics**: Ρομποτική και αυτονομία
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
    
    st.markdown('---')
    section_title('1.2 Κύρια Δομικά Στοιχεία της Τεχνητής Νοημοσύνης')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        ### 📊 Δεδομένα
        Τα δεδομένα είναι η βάση κάθε AI συστήματος
        - Big Data
        - Ποιότητα δεδομένων
        - Προεπεξεργασία
        """)
    with col2:
        st.markdown("""
        ### ⚙️ Αλγόριθμοι
        Μαθηματικές τεχνικές και μεθοδολογίες
        - Supervised Learning
        - Unsupervised Learning
        - Reinforcement Learning
        """)
    with col3:
        st.markdown("""
        ### 🎯 Μοντέλα
        Εκπαιδευμένα συστήματα
        - Neural Networks
        - Decision Trees
        - SVM
        """)
    with col4:
        st.markdown("""
        ### 💻 Υποδομές
        Υπολογιστική ισχύς
        - GPU/TPU
        - Cloud Computing
        - Frameworks
        """)
    
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
    """)
    
    exercise_choice = st.selectbox('🎯 Επιλέξτε Άσκηση:', [
        'Πρόβλεψη Τιμών (Regression)',
        'Ταξινόμηση Εικόνων (Image Classification Simulation)',
        'Sentiment Analysis Simulator',
        'Δημιουργία Συστήματος Συστάσεων'
    ])
    
    if exercise_choice == 'Πρόβλεψη Τιμών (Regression)':
        st.markdown('### 🏠 Πρόβλεψη Τιμών Ακινήτων')
        
        st.markdown("""
        Σε αυτή την άσκηση θα δημιουργήσετε ένα μοντέλο που προβλέπει την τιμή ενός ακινήτου 
        βάσει των χαρακτηριστικών του.
        """)
        
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
        
        st.dataframe(df_houses.head(10), use_container_width=True)
        
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
            with col_h2:
                st.metric('💰 Mean Absolute Error', f'{mae:,.0f} €')
            
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
