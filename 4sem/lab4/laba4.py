import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTableView, QFileDialog, QMessageBox,
    QCheckBox, QSpinBox, QGroupBox, QFormLayout, QScrollArea, QTextEdit, QComboBox)
from PyQt5.QtCore import QAbstractTableModel, Qt
import datetime


def introduce_missing_values(df, column, percent, min_group_size=2, max_group_size=10):
    """
    Вставляет пропуски в столбец column пачками случайной длины (от 2 до 10 подряд).
    
    Parameters:
        df (pd.DataFrame): исходный датафрейм
        column (str): имя столбца, в который вносятся пропуски
        percent (float): доля пропусков от общего числа строк
        min_group_size (int): минимальный размер группы пропусков
        max_group_size (int): максимальный размер группы пропусков
    
    Returns:
        pd.DataFrame: копия датафрейма с пропусками
        list: индексы всех строк с пропусками
    """
    df_copy = df.copy()
    total_missing = int(len(df) * percent / 100)
    indices_with_nans = set()
    used_indices = set()
    
    while len(indices_with_nans) < total_missing:
        group_size = np.random.randint(min_group_size, max_group_size + 1)
        start = np.random.randint(0, len(df) - group_size + 1)
        group = set(range(start, start + group_size))
        
        if used_indices.intersection(group):
            continue  # избежать перекрытия
        
        used_indices.update(group)
        indices_with_nans.update(group)

    # ограничиваем количество пропусков до нужного total_missing
    indices_with_nans = list(indices_with_nans)[:total_missing]
    df_copy.loc[indices_with_nans, column] = np.nan
    return df_copy, indices_with_nans

def impute_by_group(df, target_col, group_col):
    overall_median = df[target_col].median()
    df[target_col] = df.groupby(group_col)[target_col].transform(
        lambda x: x.fillna(x.median())
    )
    df[target_col] = df[target_col].fillna(overall_median)


def impute_by_median(df, column):
    median_val = df[column].median()
    if pd.isna(median_val):
        median_val = 0 
    df[column] = df[column].fillna(median_val)


def impute_by_zet(df, column):
    mean_val = df[column].mean()
    std_val = df[column].std()

    if pd.isna(std_val) or std_val == 0:
        std_val = 1  # избежать деления на 0

    z_scores = (df[column] - mean_val) / std_val
    close_values = df[(~df[column].isnull()) & (np.abs(z_scores) < 1.5)][column]

    median_close = close_values.median()
    if pd.isna(median_close):
        median_close = df[column].median()
    if pd.isna(median_close):
        median_close = 0  # запасной план

    df[column] = df[column].fillna(median_close)


def evaluate_recovery(original_df, restored_df, mask_indices, column):
    errors = []
    for i in mask_indices:
        true_val = original_df.loc[i, column]
        predicted_val = restored_df.loc[i, column]
        if pd.notna(true_val) and pd.notna(predicted_val) and true_val != 0:
            relative_error = abs(true_val - predicted_val) / abs(true_val)
            errors.append(relative_error)
    if errors:
        delta = 100 * (sum(errors) / len(errors))
    else:
        delta = None
    return delta

def get_distribution_parameters(df, column):
    data = df[column].dropna()
    if data.empty:
        return {}
    return {
        "mean": data.mean(),
        "median": data.median(),
        "mode": data.mode().iloc[0] if not data.mode().empty else None,
        "std": data.std(),
        "var": data.var(),
        "skew": data.skew(),
        "kurtosis": data.kurtosis(),
        "min": data.min(),
        "max": data.max(),
        "iqr": data.quantile(0.75) - data.quantile(0.25)
    }

def compare_distributions(dict1, dict2):
    result = {}
    for key in dict1:
        val1 = dict1.get(key)
        val2 = dict2.get(key)
        if val1 is not None and val2 is not None:
            result[key] = val2 - val1
        else:
            result[key] = None
    return result

class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        return str(self._df.iat[index.row(), index.column()])

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("laba4")
        self.df = pd.DataFrame()
        self.column_checkboxes = {}
        self.init_ui()
        self.df_with_missing = pd.DataFrame()
        self.miss_indices = {}

    def init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        self.setCentralWidget(central)

        load_btn = QPushButton("Загрузить xml")
        load_btn.clicked.connect(self.load_data)
        main_layout.addWidget(load_btn)

        self.table = QTableView()
        main_layout.addWidget(self.table)

        self.columns_box = QGroupBox("Столбцы для работы")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.columns_box)
        self.columns_layout = QVBoxLayout(self.columns_box)
        main_layout.addWidget(scroll)

        form_layout = QFormLayout()
        self.percent_sb = QSpinBox()
        self.percent_sb.setRange(1, 100)
        self.percent_sb.setValue(10)
        form_layout.addRow("Процент затирания:", self.percent_sb)

        self.group_cb = QCheckBox("Групповая медиана")
        self.median_cb = QCheckBox("Глобальная медиана")
        self.zscore_cb = QCheckBox("Zet-алгоритм")

        for cb in [self.group_cb, self.median_cb, self.zscore_cb]:
            cb.setChecked(True)
            form_layout.addRow(cb)

        self.group_selector = QComboBox()
        form_layout.addRow("Столбец для группировки:", self.group_selector)

        erase_btn = QPushButton("Затереть")
        erase_btn.clicked.connect(self.introduce_missing_data)
        form_layout.addRow(erase_btn)

        impute_btn = QPushButton("Восстановить")
        impute_btn.clicked.connect(self.run_imputation)
        form_layout.addRow(impute_btn)

        main_layout.addLayout(form_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        main_layout.addWidget(self.log)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Excel File', '', 'Excel Files (*.xlsx *.xls)')
        if not path:
            return
        try:
            self.df = pd.read_excel(path)
            self.table.setModel(PandasModel(self.df))
            self.update_checkboxes()
            self.update_group_selector()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")

    def update_checkboxes(self):
        for i in reversed(range(self.columns_layout.count())):
            self.columns_layout.itemAt(i).widget().setParent(None)

        self.column_checkboxes = {}
        for col in self.df.columns:
            cb = QCheckBox(col)
            cb.setChecked(True)
            self.columns_layout.addWidget(cb)
            self.column_checkboxes[col] = cb

    def update_group_selector(self):
        self.group_selector.clear()
        self.group_selector.addItems(self.df.columns.astype(str).tolist())

    def introduce_missing_data(self):
        if self.df.empty:
            QMessageBox.warning(self, "Warning", "Сначала загрузите файл.")
            return

        selected_cols = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]
        percent = self.percent_sb.value()

        self.df_with_missing = self.df.copy()
        self.miss_indices = {}
        for col in selected_cols:
            self.df_with_missing, indices = introduce_missing_values(self.df_with_missing, col, percent)
            self.miss_indices[col] = indices

        self.table.setModel(PandasModel(self.df_with_missing))
        self.log.append(f"Затёрто {percent}% значений в колонках: {', '.join(selected_cols)}")


    def run_imputation(self):
        if self.df_with_missing.empty or not self.miss_indices:
            QMessageBox.warning(self, "Warning", "Сначала выполните затирание.")
            return

        selected_cols = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]
        group_col = self.group_selector.currentText()
        methods = []
        if self.group_cb.isChecked():
            methods.append("group")
        if self.median_cb.isChecked():
            methods.append("median")
        if self.zscore_cb.isChecked():
            methods.append("zscore")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries = [f"--- Восстановление начато {timestamp} ---"]

        for col in selected_cols:
            if col not in self.miss_indices:
                continue
            for method in methods:
                df_copy = self.df_with_missing.copy()
                try:
                    if method == "group":
                        if not group_col or group_col == col:
                            raise ValueError("Неверный столбец для группировки.")
                        impute_by_group(df_copy, col, group_col)
                    elif method == "median":
                        impute_by_median(df_copy, col)
                    elif method == "zscore":
                        impute_by_zet(df_copy, col)
                    recovery = evaluate_recovery(self.df, df_copy, self.miss_indices[col], col)
                    diff = compare_distributions(get_distribution_parameters(self.df, col), get_distribution_parameters(df_copy, col))
                    if recovery is not None:
                        log_entries.append(f"{method} | {col} | ошибка восстановления: {recovery:.2f}%\nDist diff: {diff}")
                    else:
                        log_entries.append(f"{method} | {col} | ошибка восстановления: N/A\nDist diff: {diff}")
                    self.table.setModel(PandasModel(df_copy))
                except Exception as e:
                    log_entries.append(f"Ошибка в методе {method} для колонки {col}: {e}")

        log_entries.append("--- Восстановление завершено ---")
        log_text = "\n".join(log_entries)
        self.log.append(log_text)
        with open("imputation_log.txt", "a", encoding="utf-8") as f:
            f.write(log_text + "\n\n")

        selected_cols = [col for col, cb in self.column_checkboxes.items() if cb.isChecked()]
        percent = self.percent_sb.value()
        group_col = self.group_selector.currentText()
        methods = []
        if self.group_cb.isChecked():
            methods.append("group")
        if self.median_cb.isChecked():
            methods.append("median")
        if self.zscore_cb.isChecked():
            methods.append("zscore")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries = [f"--- Run started at {timestamp} ---"]

        for col in selected_cols:
            df_miss, miss_indices = introduce_missing_values(self.df, col, percent)
            log_entries.append(f"Introduced {percent}% missing in column '{col}'")
            for method in methods:
                df_copy = df_miss.copy()
                try:
                    if method == "group":
                        if not group_col or group_col == col:
                            raise ValueError("Invalid group column selected.")
                        impute_by_group(df_copy, col, group_col)
                    elif method == "median":
                        impute_by_median(df_copy, col)
                    elif method == "zscore":
                        impute_by_zet(df_copy, col)
                    recovery = evaluate_recovery(self.df, df_copy, miss_indices, col)
                    diff = compare_distributions(get_distribution_parameters(self.df, col), get_distribution_parameters(df_copy, col))
                    if recovery is not None:
                        log_entries.append(f"{percent}% missing | {method} | {col} | recovery error: {recovery:.2f}%\nDistribution diff: {diff}\n")
                    else:
                        log_entries.append(f"{percent}% missing | {method} | {col} | recovery error: N/A\nDistribution diff: {diff}\n")
                    self.table.setModel(PandasModel(df_copy))
                except Exception as e:
                    log_entries.append(f"Error with method {method} on column {col}: {e}")

        log_entries.append("--- Run complete ---")
        log_text = "\n".join(log_entries)
        self.log.append(log_text)
        with open("imputation_log.txt", "a", encoding="utf-8") as f:
            f.write(log_text + "\n\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
