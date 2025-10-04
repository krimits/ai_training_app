import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
st.set_page_config(layout='wide', page_title='AI Training — Εισαγωγή')

def section_title(t): st.markdown(f'## {t}')

def show_quiz(q):
    st.write('**Ερώτηση:**', q['question'])
    if q['type'] == 'mcq':
        choice = st.radio('Επιλογές:', q['options'], key=q['id'])
        if st.button('Υποβολή απάντησης', key=q['id']+'_btn'):
            if choice == q['answer']:
                st.success('Σωστό! ' + q.get('explain',''))
            else:
                st.error('Λάθος. ' + q.get('explain',''))
    elif q['type'] == 'tf':
        choice = st.radio('', ['Σωστό','Λάθος'], key=q['id'])
        if st.button('Υποβολή', key=q['id']+'_btn'):
            ans = 'Σωστό' if q['answer']==True else 'Λάθος'
            if choice==ans:
                st.success('Σωστό! ' + q.get('explain',''))
            else:
                st.error('Λάθος. ' + q.get('explain',''))

st.title('AI Training — Γενικά για Τεχνητή Νοημοσύνη (Ενότητες 1.1–1.7)')

tabs = st.tabs(['Περιεχόμενο','Παραδείγματα Python','Εξομοιώσεις AI','Κουίζ','Πόροι & Οδηγίες'])

with tabs[0]:
    section_title('1.1 Εισαγωγή — Τι είναι η Τεχνητή Νοημοσύνη')
    st.write("""Η Τεχνητή Νοημοσύνη (AI) είναι ο κλάδος που ασχολείται με την ανάπτυξη συστημάτων
    που εκτελούν εργασίες που απαιτούν νοημοσύνη. Βασικές έννοιες: ML, Deep Learning, NN, NLP.""")
    section_title('1.2 Κύρια δομικά στοιχεία της AI')
    st.write('- Δεδομένα, Αλγόριθμοι, Μοντέλα, Υποδομές')
    section_title('1.3 Ιστορικά επιτεύγματα')
    st.write('- Από το 1950 έως σήμερα: συμβολή αλγορίθμων και υπολογιστικής ισχύος.')
    section_title('1.4-1.7 Βασικές αρχές & ChatGPT (σύντομη περίληψη)')
    st.write('Μηχανική μάθηση, βαθιά μάθηση, Generative AI, Εφαρμογές σε τομείς. Σύντομη περιγραφή του ChatGPT ως transformer-based language model.')

with tabs[1]:
    section_title('Παράδειγμα 1: Logistic Regression (Επιβλεπόμενη Μάθηση)')
    st.write('Δημιουργούμε συνθετικά δεδομένα, εκπαιδεύουμε LogisticRegression και εμφανίζουμε αποτελέσματα.')
    n = st.slider('Αριθμός δειγμάτων', 100, 2000, 500)
    X, y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=['x1','x2']); df['y']=y
    st.dataframe(df.head(10))
    if st.button('Εκπαίδευση Logistic Regression'):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)
        model = LogisticRegression().fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.write('Accuracy:', round(acc,4))
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=X_test[:,0], y=X_test[:,1], hue=preds, palette='deep', ax=ax)
        ax.set_title('Πρόβλεψη (LogisticRegression)')
        st.pyplot(fig)
        cm = confusion_matrix(y_test, preds)
        st.write('Confusion Matrix:')
        st.table(cm)

    section_title('Παράδειγμα 2: K-Means Clustering (Μη Επιβλεπόμενη Μάθηση)')
    n_clusters = st.slider('clusters', 2,5,3,key='kc')
    Xc, yc = make_blobs(n_samples=400, centers=n_clusters, n_features=2, random_state=42)
    if st.button('Εκτέλεση K-Means'):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xc)
        labels = km.labels_
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=Xc[:,0], y=Xc[:,1], hue=labels, palette='tab10', ax=ax)
        ax.set_title('KMeans clustering')
        st.pyplot(fig)

with tabs[2]:
    section_title('Εξομοιώσεις AI — Απλό demo')
    st.write('Εδώ βλέπετε πώς αλλάζει ένα μοντέλο όταν αλλάζουμε παραμέτρους.')
    noise = st.slider('Παράμετρος θορύβου (noise)', 0.0, 2.0, 0.5)
    Xn, yn = make_classification(n_samples=400, n_features=2, n_redundant=0, flip_y=noise*0.1, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=0.3, random_state=0)
    model = LogisticRegression().fit(Xtr, ytr)
    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)
    st.write('Accuracy with noise', round(acc,3))
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=Xte[:,0], y=Xte[:,1], hue=preds, ax=ax)
    st.pyplot(fig)
    st.write('Σημείωση: Αυτές οι εξομοιώσεις είναι εκπαιδευτικού χαρακτήρα.')

with tabs[3]:
    section_title('Κουίζ αυτοαξιολόγησης')
    quizzes = [
        {'id':'q1','type':'mcq','question':'Τι είναι μηχανική μάθηση;','options':['Γραφικά υπολογιστών','Υποκατηγορία της AI που μαθαίνει από δεδομένα','Μόνο δίκτυα νευρώνων'],'answer':'Υποκατηγορία της AI που μαθαίνει από δεδομένα','explain':'Η ML είναι η υποκατηγορία της AI που επιτρέπει σε αλγορίθμους να μαθαίνουν από δεδομένα.'},
        {'id':'q2','type':'tf','question':'Το ChatGPT είναι transformer-based language model.','answer':True,'explain':'Σωστά — το ChatGPT βασίζεται στην αρχιτεκτονική Transformer.'}
    ]
    for q in quizzes:
        show_quiz(q)

with tabs[4]:
    section_title('Πόροι & Οδηγίες')
    st.write('README βρίσκεται στο φάκελο της εφαρμογής. Για offline χρήση, εγκαταστήστε απαιτήσεις τοπικά.')
