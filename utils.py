import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(base_addr='Data/'):
    df_x_train = pd.read_csv(os.path.join(base_addr, 'arguments-training.tsv'), sep='\t')
    df_x_val = pd.read_csv(os.path.join(base_addr, 'arguments-validation.tsv'), sep='\t')
    df_x_test = pd.read_csv(os.path.join(base_addr, 'arguments-test.tsv'), sep='\t')
    df_y_train = pd.read_csv(os.path.join(base_addr, 'labels-training.tsv'), sep='\t')
    df_y_val = pd.read_csv(os.path.join(base_addr, 'labels-validation.tsv'), sep='\t')
    return df_x_train, df_y_train, df_x_val, df_y_val, df_x_test

def evaluate(name, y_pred, y_true):
    class_name = ['Self-direction: thought', 'Self-direction: action',
       'Stimulation', 'Hedonism', 'Achievement', 'Power: dominance',
       'Power: resources', 'Face', 'Security: personal', 'Security: societal',
       'Tradition', 'Conformity: rules', 'Conformity: interpersonal',
       'Humility', 'Benevolence: caring', 'Benevolence: dependability',
       'Universalism: concern', 'Universalism: nature',
       'Universalism: tolerance', 'Universalism: objectivity']
    class_dict = {}
    for c, i in enumerate(class_name):
        class_dict[c] = i
    res = []
    columns = ["f1-macro","f1-micro","f1-weighted"]
    os.makedirs(os.path.join(name), exist_ok=True)
    res.append(metrics.f1_score(y_true, y_pred, average='macro'))
    res.append(metrics.f1_score(y_true, y_pred, average='micro'))
    res.append(metrics.f1_score(y_true, y_pred, average='weighted'))
    for c, i in enumerate(metrics.f1_score(y_true, y_pred, average=None)):
        res.append(i)
        columns.append(f'f1 on class {class_dict[c]}')

    df = pd.DataFrame(res, columns)
    with open(os.path.join(name,'res.tex'), 'w') as f:
        f.write(df.to_latex())

    matrices = metrics.multilabel_confusion_matrix(y_true, y_pred)
    
    for c, i in enumerate(matrices):
        cmd = metrics.ConfusionMatrixDisplay(i, display_labels=np.unique(y_true)).plot()
        plt.title(f'Confusion Matrix for label {class_dict[c]}')
        plt.savefig(os.path.join(name,f'Confusion_{class_dict[c]}.png'))

    return df

def get_emb(name, text):
    if name=='tf-idf':
        all_text = np.concatenate([text['train'], text['test']])
        vectorizer = TfidfVectorizer()
        emb = vectorizer.fit_transform(all_text)
        return emb[:text['train'].shape[0]], emb[text['train'].shape[0]:]
    elif name=='labse':
        labse_model = SentenceTransformer('sentence-transformers/LaBSE')
        emb = labse_model.encode(text)
        return emb
