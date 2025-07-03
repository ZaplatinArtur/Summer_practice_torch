import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

class ClassificationMetrics:
    def __init__(self):
        """
        Инициализация класса метрик классификации
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_one_hot(self, labels, num_classes):
        """
        Преобразование меток в one-hot encoding
        
        Args:
            labels: тензор меток (batch_size,)
            num_classes: количество классов
            
        Returns:
            one-hot тензор (batch_size, num_classes)
        """
        return torch.nn.functional.one_hot(labels, num_classes).float()

    def precision(self, y_true, y_pred, average='macro'):
        """
        Вычисление метрики Precision
        
        Args:
            y_true: истинные метки (batch_size,)
            y_pred: предсказанные метки (batch_size,)
            average: тип усреднения ('macro', 'micro', 'weighted')
            
        Returns:
            precision score
        """
        y_true = torch.tensor(y_true, dtype=torch.long).to(self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)
        
        num_classes = len(torch.unique(y_true))
        
        if average == 'micro':
            true_positives = torch.sum(y_true == y_pred).float()
            predicted_positives = len(y_pred)
            return true_positives / predicted_positives if predicted_positives > 0 else 0
        
        precision_per_class = []
        weights = []
        
        for c in range(num_classes):
            true_positives = torch.sum((y_true == c) & (y_pred == c)).float()
            predicted_positives = torch.sum(y_pred == c).float()
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            precision_per_class.append(precision)
            
            if average == 'weighted':
                weights.append(torch.sum(y_true == c).float() / len(y_true))
        
        precision_per_class = torch.tensor(precision_per_class)
        
        if average == 'macro':
            return torch.mean(precision_per_class).item()
        elif average == 'weighted':
            return torch.sum(precision_per_class * torch.tensor(weights)).item()
        
        return precision_per_class.tolist()

    def recall(self, y_true, y_pred, average='macro'):
        """
        Вычисление метрики Recall
        
        Args:
            y_true: истинные метки (batch_size,)
            y_pred: предсказанные метки (batch_size,)
            average: тип усреднения ('macro', 'micro', 'weighted')
            
        Returns:
            recall score
        """
        y_true = torch.tensor(y_true, dtype=torch.long).to(self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)
        
        num_classes = len(torch.unique(y_true))
        
        if average == 'micro':
            true_positives = torch.sum(y_true == y_pred).float()
            actual_positives = len(y_true)
            return true_positives / actual_positives if actual_positives > 0 else 0
        
        recall_per_class = []
        weights = []
        
        for c in range(num_classes):
            true_positives = torch.sum((y_true == c) & (y_pred == c)).float()
            actual_positives = torch.sum(y_true == c).float()
            
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            recall_per_class.append(recall)
            
            if average == 'weighted':
                weights.append(torch.sum(y_true == c).float() / len(y_true))
        
        recall_per_class = torch.tensor(recall_per_class)
        
        if average == 'macro':
            return torch.mean(recall_per_class).item()
        elif average == 'weighted':
            return torch.sum(recall_per_class * torch.tensor(weights)).item()
        
        return recall_per_class.tolist()

    def f1_score(self, y_true, y_pred, average='macro'):
        """
        Вычисление метрики F1-score
        
        Args:
            y_true: истинные метки (batch_size,)
            y_pred: предсказанные метки (batch_size,)
            average: тип усреднения ('macro', 'micro', 'weighted')
            
        Returns:
            f1 score
        """
        precision = self.precision(y_true, y_pred, average)
        recall = self.recall(y_true, y_pred, average)
        
        if average in ['macro', 'weighted']:
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        elif average == 'micro':
            return precision
        
        f1_scores = []
        for p, r in zip(precision, recall):
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
        return f1_scores

    def roc_auc(self, y_true, y_scores, multi_class='ovr'):
        """
        Вычисление метрики ROC-AUC
        
        Args:
            y_true: истинные метки (batch_size,)
            y_scores: вероятности предсказаний (batch_size, num_classes)
            multi_class: 'ovr' (one-vs-rest) или 'ovo' (one-vs-one)
            
        Returns:
            roc-auc score
        """
        y_true = torch.tensor(y_true, dtype=torch.long).cpu().numpy()
        y_scores = torch.tensor(y_scores, dtype=torch.float).cpu().numpy()
        
        num_classes = y_scores.shape[1]
        y_true_one_hot = self.to_one_hot(torch.tensor(y_true), num_classes).cpu().numpy()
        
        auc_scores = []
        
        if multi_class == 'ovr':
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_scores[:, i])
                auc_score = auc(fpr, tpr)
                auc_scores.append(auc_score)
            return np.mean(auc_scores)
        
        elif multi_class == 'ovo':
            auc_sum = 0
            n_pairs = 0
            for i in range(num_classes):
                for j in range(i+1, num_classes):
                    mask = np.logical_or(y_true == i, y_true == j)
                    y_true_binary = (y_true[mask] == i).astype(int)
                    y_scores_binary = y_scores[mask, i] / (y_scores[mask, i] + y_scores[mask, j])
                    fpr, tpr, _ = roc_curve(y_true_binary, y_scores_binary)
                    auc_sum += auc(fpr, tpr)
                    n_pairs += 1
            return auc_sum / n_pairs if n_pairs > 0 else 0
        
        return auc_scores

    def confusion_matrix(self, y_true, y_pred, plot=True, save_path=None):
        """
        Вычисление и визуализация confusion matrix с возможностью сохранения
        
        Args:
            y_true: истинные метки (batch_size,)
            y_pred: предсказанные метки (batch_size,)
            plot: флаг для отображения визуализации
            save_path: путь для сохранения графика (например, 'confusion_matrix.png')
            
        Returns:
            confusion matrix
        """
        y_true = torch.tensor(y_true, dtype=torch.long).to(self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)
        
        num_classes = len(torch.unique(y_true))
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        
        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_path:
                # Создаём директорию, если она не существует
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix сохранена в {save_path}")
            
            plt.show()
            plt.close()
        
        return cm



from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, path_file, numeric_columns: list, string_columns: list, binary_columns: list, target_column: str):
        self.path_file = path_file
        self.numeric_columns = numeric_columns
        self.string_columns = string_columns
        self.binary_columns = binary_columns
        self.target_column = target_column
        self.label_encoders = {}
        self.one_hot_encoders = {}

        self.df = pd.read_csv(path_file)
        self.df = self.df.dropna()

        self._convert_numeric()
        self._convert_string()
        self._convert_binary()

    def _convert_numeric(self):
        if self.numeric_columns:
            scaler = StandardScaler()
            self.df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])

    def _convert_string(self):
        if self.string_columns:
            for column in self.string_columns:
                le = LabelEncoder()
                self.df[column] = self.df[column].fillna('missing')
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le

    def _convert_binary(self):
        if self.binary_columns:
            for column in self.binary_columns:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                transformed = encoder.fit_transform(self.df[[column]])
                new_columns = [f"{column}_{cat}" for cat in encoder.categories_[0][1:]]
                self.df[new_columns] = transformed
                self.df = self.df.drop(column, axis=1)
                self.one_hot_encoders[column] = encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data_one_row = self.df.iloc[index]
        train = data_one_row.drop(self.target_column).values.astype(np.float32)  # Преобразуем в numpy float32
        target = np.array(data_one_row[self.target_column], dtype=np.float32)  # Преобразуем в numpy float32
        return torch.tensor(train), torch.tensor(target)  # Возвращаем тензоры