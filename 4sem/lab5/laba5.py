import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random

PREVIEW_ROWS = 300

def forel(X: np.ndarray, r: float):
    n = X.shape[0]
    lab = np.full(n, -1, int)
    left = np.arange(n); cid = 0
    while left.size:
        c = X[random.choice(left)]
        while True:
            d = np.linalg.norm(X[left] - c, axis=1)
            ins = left[d <= r]
            new_c = X[ins].mean(axis=0)
            if np.allclose(new_c, c): break
            c = new_c
        lab[ins] = cid
        left = left[d > r]; cid += 1
    return lab

def compactness(X: np.ndarray, labels: np.ndarray) -> float:
    return sum(((X[labels == k] - X[labels == k].mean(axis=0)) ** 2).sum()
               for k in np.unique(labels) if k != -1)

def cluster_statistics(X: np.ndarray, labels: np.ndarray):
    out = []
    for k in np.unique(labels):
        if k == -1: continue
        pts = X[labels == k]
        ctr = pts.mean(axis=0)
        out.append(dict(cluster=int(k),
                        size=int(len(pts)),
                        sse=float(((pts - ctr) ** 2).sum()),
                        rmax=float(np.linalg.norm(pts - ctr, axis=1).max())))
    return out

def feature_search(df: pd.DataFrame, r: float, k: int, max_iter=80, seed=0):
    rng = np.random.default_rng(seed)
    n = df.shape[1]; k = max(1, min(k, n))
    w = np.ones(n); best_sub, best_sse = list(range(k)), np.inf
    for _ in range(max_iter):
        sub = rng.choice(n, size=k, replace=False, p=w/w.sum())
        X = df.iloc[:, sub].to_numpy()
        lbl = forel(X, r); sse = compactness(X, lbl)
        if sse < best_sse:
            best_sse, best_sub = sse, sub
            w[sub] *= 1.3
        w *= 0.98
    return sorted(best_sub), best_sse

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat = df.select_dtypes(include=['object', 'category']).columns
    num = df.select_dtypes(include=['number']).columns

    for col in cat:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("MISSING").astype(str))

    if len(num):
        df[num] = SimpleImputer(strategy="median").fit_transform(df[num])
        df[num] = StandardScaler().fit_transform(df[num])

    if df.isna().any().any():
        raise ValueError("NaN остались после препроцессинга")

    return df.astype(float)

def anonymize_df(df: pd.DataFrame) -> pd.DataFrame:
    anon = df.copy()

    if 'VIN (1-10)' in anon:
        anon['VIN (1-10)'] = anon['VIN (1-10)'].astype(str).str[:1] + '********'

    drop_cols = [c for c in anon.columns if 'ID' in c.upper()] + [
        '2020 Census Tract', 'State'
    ]
    anon.drop(columns=[c for c in drop_cols if c in anon], inplace=True, errors='ignore')

    if 'County' in anon:
        anon['County'] = anon['County'].astype(str).str[:3] + '***'
    if 'City' in anon:
        anon['City'] = 'City_' + anon['City'].astype('category').cat.codes.astype(str)

    if 'Vehicle Location' in anon:
        anon['Vehicle Location'] = anon['Vehicle Location'].str.replace(
            r'(-?\d+\\.\\d{2})\\d+', lambda m: f"{float(m.group(1)):.1f}", regex=True)

    if 'Electric Range' in anon:
        def rng_cat(x):
            if pd.isna(x) or x == 0: return 'No'
            if x < 50: return 'Short'
            if x < 200: return 'Mid'
            return 'Long'
        anon['Electric Range'] = anon['Electric Range'].apply(rng_cat)

    for col, mod in [('Make', 10), ('Model', 20)]:
        if col in anon:
            anon[col] = (anon[col].astype(str)
                         .apply(lambda v: f"{col}_{hash(v) % mod}"))

    if 'Postal Code' in anon:
        anon['Postal Code'] = anon['Postal Code'].astype(str).str[:3]

    if 'Model Year' in anon:
        anon['Model Year'] = (anon['Model Year'] // 3 * 3).astype(str) + 's'

    num_cols = anon.select_dtypes(include=['number']).columns
    for col in num_cols:
        sigma = anon[col].std() * 0.05
        anon[col] = (anon[col] + np.random.normal(0, sigma, size=len(anon))).round(2)

    return anon

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("laba5")
        self.geometry("1320x820")

        self.raw: pd.DataFrame | None = None       
        self.anonym: pd.DataFrame | None = None    
        self.proc: pd.DataFrame | None = None     
        self.selected: list[int] | None = None
        self.labels = None
        self.r_var = tk.DoubleVar(value=1.0)
        self.k_var = tk.IntVar(value=4)
        self._build_ui()

    def _build_ui(self):
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=780)             
        left.pack_propagate(False)                     
        main.add(left, stretch="always")               

        ctrl = ttk.Frame(left); ctrl.pack(fill=tk.X, padx=5, pady=3)

        ttk.Button(ctrl, text="Загрузить CSV", command=self.load_csv)\
            .pack(fill=tk.X, pady=2)
        
        row = ttk.Frame(ctrl); row.pack(fill=tk.X)
        ttk.Label(row, text="Радиус R").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.r_var, width=7)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="K признаков").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Entry(row, textvariable=self.k_var, width=6)\
            .pack(side=tk.LEFT, padx=4)

        ttk.Button(ctrl, text="Выбрать признаки (Random Search)",
                   command=self.select_features).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Кластеризовать",
                   command=self.cluster).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Оценить компактность",
                   command=self.eval_comp).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Статистика кластеров",
                   command=self.show_stats).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Обезличить данные",
                   command=self.anonymize).pack(fill=tk.X, pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=3, padx=5)

        tv_frame = ttk.Frame(left)
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 3))

        self.tree = ttk.Treeview(tv_frame, show="headings")
        vsb = ttk.Scrollbar(tv_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(left, text="Лог").pack(anchor=tk.W, padx=5)
        self.log = tk.Text(left, height=8)
        self.log.pack(fill=tk.X, padx=5, pady=(0, 5))

        right = ttk.Frame(main, width=400)       
        main.add(right)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log_msg(self, txt: str):
        self.log.insert(tk.END, txt + "\n")
        self.log.see(tk.END)

    def update_table(self, df: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        if df is None or df.empty:
            return
        df_head = df.head(PREVIEW_ROWS)
        self.tree["columns"] = list(df_head.columns)
        for col in df_head.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=110, anchor="center")
        for _, row in df_head.iterrows():
            self.tree.insert("", "end", values=row.to_numpy())

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.raw = pd.read_csv(path)
            self.anonym = None
            self.proc = preprocess(self.raw)
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            return

        self.labels = self.selected = None
        self.update_table(self.raw)
        self.draw()
        self.log_msg(f"Загружено: {path} | строк: {len(self.raw)}")

    def anonymize(self):
        if self.raw is None:
            messagebox.showinfo("Нет данных", "Сначала загрузите CSV")
            return
        try:
            self.anonym = anonymize_df(self.raw)
            self.proc = preprocess(self.anonym)
        except Exception as e:
            messagebox.showerror("Анонимизация", str(e))
            return

        self.labels = self.selected = None
        self.update_table(self.anonym)
        self.draw()
        self.log_msg("Данные успешно обезличены")

    def select_features(self):
        if self.proc is None:
            messagebox.showinfo("Нет данных", "Загрузите или обезличьте данные")
            return
        k = max(1, min(self.k_var.get(), self.proc.shape[1]))
        self.k_var.set(k)
        sub, sse = feature_search(self.proc, self.r_var.get(), k)
        self.selected = sub
        self.labels = None
        cols = [self.proc.columns[i] for i in sub]
        self.update_table((self.anonym or self.raw)[cols])
        self.draw()
        self.log_msg(f"Random Search: выбрано {k} признаков | SSE={sse:.1f}")

    def cluster(self):
        if self.proc is None:
            messagebox.showinfo("Нет данных", "Загрузите/обезличьте данные")
            return
        cols = self.selected or list(range(self.proc.shape[1]))
        X = self.proc.iloc[:, cols].to_numpy()
        self.labels = forel(X, self.r_var.get())
        self.draw()
        self.log_msg(f"FOREL: найдено {len(np.unique(self.labels))} кластеров")

    def eval_comp(self):
        if self.labels is None:
            messagebox.showinfo("Кластеры", "Сначала выполните кластеризацию")
            return
        cols = self.selected or list(range(self.proc.shape[1]))
        score = compactness(self.proc.iloc[:, cols].to_numpy(), self.labels)
        self.log_msg(f"Компактность (SSE) = {score:.2f}")

    def show_stats(self):
        if self.labels is None:
            messagebox.showinfo("Кластеры", "Сначала выполните кластеризацию")
            return
        cols = self.selected or list(range(self.proc.shape[1]))
        stats = cluster_statistics(self.proc.iloc[:, cols].to_numpy(), self.labels)
        for s in stats:
            self.log_msg(f"id={s['cluster']}  size={s['size']}  "
                         f"SSE={s['sse']:.1f}  Rmax={s['rmax']:.2f}")

    def draw(self):
        self.ax.clear()
        if self.proc is None:
            self.ax.set_title("Нет данных")
            self.canvas.draw()
            return

        cols = self.selected or list(range(self.proc.shape[1]))
        X = self.proc.iloc[:, cols].to_numpy()

        # 2-D проекция
        if X.shape[1] > 2:
            X2 = PCA(n_components=2).fit_transform(X)
            xlabel, ylabel = "PC1", "PC2"
        else:
            X2 = X[:, :2]
            xlabel = self.proc.columns[cols[0]]
            ylabel = self.proc.columns[cols[1]] if len(cols) > 1 else ""

        # scatter
        if self.labels is not None:
            for k in np.unique(self.labels):
                idx = self.labels == k
                self.ax.scatter(X2[idx, 0], X2[idx, 1],
                                label=f"C{k}", s=40, alpha=0.8)
            self.ax.legend()
            title = "Кластеры (FOREL)"
        else:
            self.ax.scatter(X2[:, 0], X2[:, 1], color="gray", s=25, alpha=0.6)
            title = "Данные / признаки"

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.fig.tight_layout()
        self.canvas.draw()

    def __init__(self):
        super().__init__()
        self.title('5 laba')
        self.geometry("1320x820")

        # datasets
        self.raw:  pd.DataFrame | None = None  # оригинал
        self.anonym: pd.DataFrame | None = None  # обезличенный
        self.proc: pd.DataFrame | None = None  # текущий для ML
        self.labels = None
        self.selected: list[int] | None = None

        # vars
        self.r_var = tk.DoubleVar(value=1.0)
        self.k_var = tk.IntVar(value=4)

        self._build()

    def _build(self):
        main = ttk.PanedWindow(self, orient="horizontal"); main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=380); left.pack_propagate(False); main.add(left)

        ttk.Button(left, text="Загрузить CSV", command=self.load_csv).pack(fill=tk.X, padx=5, pady=3)

        row = ttk.Frame(left); row.pack(fill=tk.X, padx=5)
        ttk.Label(row, text="Радиус R").pack(side=tk.LEFT); ttk.Entry(row, textvariable=self.r_var, width=8).pack(side=tk.LEFT, padx=3)

        row2 = ttk.Frame(left); row2.pack(fill=tk.X, padx=5)
        ttk.Label(row2, text="K признаков").pack(side=tk.LEFT); ttk.Entry(row2, textvariable=self.k_var, width=6).pack(side=tk.LEFT, padx=3)

        ttk.Button(left, text="Выбрать признаки (Random Search)", command=self.select_features).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(left, text="Кластеризовать FOREL", command=self.cluster).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(left, text="Оценить компактность", command=self.eval_comp).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(left, text="Статистика кластеров", command=self.show_stats).pack(fill=tk.X, padx=5, pady=3)
        ttk.Button(left, text="Обезличить данные", command=self.anonymize).pack(fill=tk.X, padx=5, pady=6)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=4, padx=5)

        tvf = ttk.Frame(left); tvf.pack(fill=tk.BOTH, expand=True, padx=5)
        self.tree = ttk.Treeview(tvf, show="headings")
        ysb = ttk.Scrollbar(tvf, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(tvf, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(left, text="Лог").pack(anchor=tk.W, padx=5)
        self.log = tk.Text(left, height=9); self.log.pack(fill=tk.X, padx=5, pady=(0,5))

        # right plot
        right = ttk.Frame(main); main.add(right)
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log_msg(self, t): self.log.insert(tk.END, t+"\n"); self.log.see(tk.END)

    def load_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if not p: return
        try:
            self.raw = pd.read_csv(p)
            self.anonym = None
            self.proc = preprocess(self.raw)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e)); return
        self.labels = self.selected = None
        self.update_table(self.raw)
        self.draw()
        self.log_msg(f"Загружено: {p} | строк: {len(self.raw)}")

    def update_table(self, df):
        self.tree.delete(*self.tree.get_children())
        if df is None or df.empty: return
        df_head = df.head(PREVIEW_ROWS)
        self.tree["columns"] = list(df_head.columns)
        for c in df_head.columns:
            self.tree.heading(c, text=c); self.tree.column(c, width=110, anchor="center")
        for _, row in df_head.iterrows():
            self.tree.insert("", "end", values=row.to_numpy())

    def anonymize(self):
        if self.raw is None:
            messagebox.showinfo("Данные", "Сначала загрузите CSV"); return
        try:
            self.anonym = anonymize_df(self.raw)
            self.proc = preprocess(self.anonym)
        except Exception as e:
            messagebox.showerror("Анонимизация", str(e)); return
        self.labels = self.selected = None
        self.update_table(self.anonym)
        self.draw()
        self.log_msg("Данные обезличены; дальнейшие операции идут по анонимизированному набору")

    def select_features(self):
        if self.proc is None:
            messagebox.showinfo("Нет данных", "Загрузите (и/или обезличьте) данные"); return
        k = min(max(self.k_var.get(),1), self.proc.shape[1]); self.k_var.set(k)
        sub, sse = feature_search(self.proc, self.r_var.get(), k)
        self.selected, self.labels = sub, None
        cols = [self.proc.columns[i] for i in sub]
        self.log_msg(f"K={k} признаков выбрано | SSE={sse:.1f}")
        self.update_table((self.anonym or self.raw)[cols])
        self.draw()

    def cluster(self):
        if self.proc is None:
            messagebox.showinfo("Нет данных", "Загрузите (и/или обезличьте) данные"); return
        X = self.proc.iloc[:, self.selected or list(range(self.proc.shape[1]))].to_numpy()
        self.labels = forel(X, self.r_var.get())
        self.log_msg(f"FOREL: найдено {len(np.unique(self.labels))} кластеров")
        self.draw()

    def eval_comp(self):
        if self.labels is None:
            messagebox.showinfo("Кластеры", "Сначала кластеризуйте"); return
        X = self.proc.iloc[:, self.selected or list(range(self.proc.shape[1]))].to_numpy()
        self.log_msg(f"Компактность (SSE): {compactness(X, self.labels):.2f}")

    def show_stats(self):
        if self.labels is None:
            messagebox.showinfo("Кластеры", "Сначала кластеризуйте"); return
        X = self.proc.iloc[:, self.selected or list(range(self.proc.shape[1]))].to_numpy()
        for s in cluster_statistics(X, self.labels):
            self.log_msg(f"id={s['cluster']} size={s['size']} SSE={s['sse']:.1f} Rmax={s['rmax']:.2f}")

    def draw(self):
        self.ax.clear()
        if self.proc is None:
            self.ax.set_title("Нет данных"); self.canvas.draw(); return

        cols = self.selected or list(range(self.proc.shape[1]))
        X = self.proc.iloc[:, cols].to_numpy()
        if X.shape[1] > 2:
            X2 = PCA(n_components=2).fit_transform(X); xl,yl="PC1","PC2"
        else:
            X2 = X[:,:2]; xl=self.proc.columns[cols[0]]; yl=self.proc.columns[cols[1]] if len(cols)>1 else ""

        if self.labels is not None:
            for k in np.unique(self.labels):
                idx = self.labels==k
                self.ax.scatter(X2[idx,0], X2[idx,1], label=f"C{k}", s=40, alpha=.8)
            self.ax.legend(); title="Кластеры"
        else:
            self.ax.scatter(X2[:,0], X2[:,1], color="gray", s=30); title="Данные/признаки"

        self.ax.set_title(title); self.ax.set_xlabel(xl); self.ax.set_ylabel(yl)
        self.fig.tight_layout(); self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()
