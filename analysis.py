import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Korelasyonu yüksek sütunları kaldırmak için fonksiyon
def remove_highly_correlated_features(data, threshold=0.95):
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return data.drop(columns=to_drop)

# 1. Tüm CSV dosyalarının bulunduğu klasörü tanımla
folder_path = './csv_files'  # CSV dosyalarınızın bulunduğu klasör yolu
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 2. Model tanımlamaları
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(),
    # 'XGBoost': XGBClassifier(scale_pos_weight=1)
    # 'SVM': SVC(random_state=42, class_weight='balanced'),
}

# 3. Sonuçları toplamak için boş bir liste oluştur
results = []

# 4. Tüm dosyalar üzerinde dön
for file in csv_files:
    print(f"\nProcessing file: {file}")
    
    # 1. Veri Yükleme
    df = pd.read_csv(os.path.join(folder_path, file))
    df.columns = df.columns.str.strip()

    # 2. Temizlik: Sonsuz, sabit, boş ve yinelenen sütunları kaldır
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, df.nunique() > 1]  # Sabit sütunları kaldır
    df = df.loc[:, ~df.T.duplicated()]  # Yinelenen sütunları kaldır
    df.dropna(inplace=True)

    # 3. Label sütununu sayısallaştır
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

    # 4. Korelasyonu yüksek sütunları kaldır
    X_temp = df.drop('Label', axis=1)
    X_temp = remove_highly_correlated_features(X_temp)

    # 5. Özellik ölçekleme
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_temp)
    X = pd.DataFrame(X_scaled, columns=X_temp.columns)
    y = df['Label']

    # 6. Eğitim/test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Her modeli eğit ve değerlendirme
    for name, model in models.items():
        print(f"Model: {name}")
        
        # Modeli eğit
        model.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred = model.predict(X_test)
        
        # Performans ölç
        classification_rep = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        
        # Sonuçları listeye ekle
        results.append({
            'File': file,
            'Model': name,
            'Accuracy': classification_rep['accuracy'],
            'Macro avg Precision': classification_rep['macro avg']['precision'],
            'Macro avg Recall': classification_rep['macro avg']['recall'],
            'Macro avg F1-Score': classification_rep['macro avg']['f1-score']
        })
        
        # Karmaşıklık matrisi
        conf_mat = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix: {name} for {file}")
        plt.show()

# 8. Sonuçları bir DataFrame'e dönüştür ve yazdır
results_df = pd.DataFrame(results)
print("\nModel Karşılaştırma Sonuçları:")
print(results_df)
