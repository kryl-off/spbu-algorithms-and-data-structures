"""
FOREL Clustering Laboratory GUI — версия 8
Новое:
  • Чекбокс «Кодировать категориальные» (по-умолчанию ✓).
  • При загрузке CSV:
       ─ числовые столбцы остаются как есть;
       ─ категориальные → pd.get_dummies(drop_first=True, dtype=float);
       ─ конкатенация => полностью числовой датасет.
  • В логе выводится, сколько dummy-столбцов создано.
Остальной функционал v7 (генерация, выбор признаков, FOREL, оценка) без изменений.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PREVIEW_ROWS = 300


# ---------- FOREL ----------------------------------------------------------
def forel(data: np.ndarray, radius: float):
    n = data.shape[0]
    labels = np.full(n, -1, int)
    remaining = np.arange(n)
    cid = 0
    while remaining.size:
        center = data[np.random.choice(remaining)]
        while True:
            d = np.linalg.norm(data[remaining] - center, axis=1)
            inside = remaining[d <= radius]
            new_center = data[inside].mean(axis=0)
            if np.allclose(new_center, center):
                break
            center = new_center
        labels[inside] = cid
        remaining = remaining[d > radius]
        cid += 1
    return labels, [data[labels == k].mean(axis=0) for k in range(cid)]


# ---------- METRICS --------------------------------------------------------
def cluster_compactness(X: np.ndarray, labels: np.ndarray):
    return sum(((X[labels == k] - X[labels == k].mean(axis=0)) ** 2).sum()
               for k in np.unique(labels) if k != -1)


def cluster_statistics(X: np.ndarray, labels: np.ndarray):
    stats = []
    for k in np.unique(labels):
        if k == -1:
            continue
        pts = X[labels == k]
        ctr = pts.mean(axis=0)
        stats.append(dict(cluster=int(k),
                          size=int(len(pts)),
                          sse=float(((pts - ctr) ** 2).sum()),
                          radius=float(np.linalg.norm(pts - ctr, axis=1).max())))
    return stats


# ---------- FEATURE SEARCH -------------------------------------------------
def feature_search(df: pd.DataFrame, radius: float, k: int,
                   max_iter: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = df.shape[1]
    k = max(1, min(k, n))
    w = np.ones(n)
    best, best_score = list(range(k)), np.inf
    for _ in range(max_iter):
        p = w / w.sum()
        sub = rng.choice(n, size=k, replace=False, p=p)
        X = df.iloc[:, sub].to_numpy(float)
        lbl, _ = forel(X, radius)
        score = cluster_compactness(X, lbl)
        if score < best_score:
            best, best_score = sub, score
            w[sub] *= 1.2
        w *= 0.99
    return sorted(best), best_score


# ---------- GUI ------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FOREL Clustering Lab v8 — кодирование категориальных")
        self.geometry("1240x780")

        # данные
        self.data: pd.DataFrame | None = None
        self.selected: list[int] | None = None
        self.labels = None

        # vars
        self.encode_var = tk.BooleanVar(value=True)
        self.n_rows = tk.IntVar(value=200)
        self.radius = tk.DoubleVar(value=1.0)
        self.k_var  = tk.IntVar(value=4)

        self._build_ui()

    # ---------- Build UI ----------------------------------------------------
    def _build_ui(self):
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill=tk.BOTH, expand=True)

        # LEFT
        left = ttk.Frame(main, width=350)
        left.pack_propagate(False)
        main.add(left)

        ctrl = ttk.Frame(left)
        ctrl.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(ctrl, text="Загрузить CSV", command=self.load_csv).pack(fill=tk.X, pady=2)

        ttk.Label(ctrl, text="N строк (генерация)").pack(anchor=tk.W)
        ttk.Entry(ctrl, textvariable=self.n_rows).pack(fill=tk.X, pady=1)
        ttk.Button(ctrl, text="Сгенерировать датасет", command=self.generate_dataset).pack(fill=tk.X, pady=2)

        ttk.Button(ctrl, text="Сохранить CSV", command=self.save_csv).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(ctrl, text="Кодировать категориальные",
                        variable=self.encode_var).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=4)

        ttk.Label(ctrl, text="Радиус R").pack(anchor=tk.W)
        ttk.Entry(ctrl, textvariable=self.radius).pack(fill=tk.X, pady=1)

        ttk.Label(ctrl, text="Количество признаков K").pack(anchor=tk.W)
        ttk.Entry(ctrl, textvariable=self.k_var).pack(fill=tk.X, pady=1)

        ttk.Button(ctrl, text="Выбрать признаки", command=self.select_features).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Кластеризовать", command=self.cluster).pack(fill=tk.X, pady=2)
        ttk.Button(ctrl, text="Оценить кластеры", command=self.evaluate_clusters).pack(fill=tk.X, pady=2)

        ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, pady=4)

        tv_frame = ttk.Frame(left)
        tv_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        self.tree = ttk.Treeview(tv_frame, show="headings")
        ysb = ttk.Scrollbar(tv_frame, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(tv_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=ysb.set, xscroll=xsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        xsb.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(left, text="Лог").pack(anchor=tk.W, padx=5)
        self.log = tk.Text(left, height=10, width=40)
        self.log.pack(fill=tk.X, padx=5, pady=(0, 5))

        # RIGHT – plot
        right = ttk.Frame(main)
        main.add(right, stretch="always")
        self.fig, self.ax = plt.subplots(figsize=(7.5, 7.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Dataset view -----------------------------------------------
    def update_dataset_view(self):
        self.tree.delete(*self.tree.get_children())
        if self.data is None:
            return
        idx = self.selected or list(range(self.data.shape[1]))
        df = self.data.iloc[:, idx].copy()
        if self.labels is not None:
            df["cluster"] = self.labels
        df = df.head(PREVIEW_ROWS)

        self.tree["columns"] = list(df.columns)
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        for _, row in df.iterrows():
            self.tree.insert("", "end", values=row.to_numpy())

    # ---------- Data operations --------------------------------------------
    def _after_load(self, df: pd.DataFrame, msg: str):
        self.data, self.labels = df, None
        self.selected = list(range(df.shape[1]))
        self._log(msg)
        self._draw_plot()
        self.update_dataset_view()

    def generate_dataset(self):
        n = self.n_rows.get()
        rng = np.random.default_rng()
        df = pd.DataFrame(rng.normal(size=(n, 16)),
                          columns=[f"f{i+1}" for i in range(16)])
        self._after_load(df, f"Сгенерирован датасет: {n}×16")

    def load_csv(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not fp:
            return
        try:
            raw = pd.read_csv(fp)
        except Exception as e:
            messagebox.showerror("Ошибка чтения", str(e))
            return

        num = raw.select_dtypes(include=[np.number])
        cat = raw.select_dtypes(exclude=[np.number])

        if self.encode_var.get() and not cat.empty:
            dummies = pd.get_dummies(cat, drop_first=True, dtype=float)
            df = pd.concat([num, dummies], axis=1)
            self._after_load(df,
                             f"Загружено: {fp}  "
                             f"(числовых: {num.shape[1]}, dummy: {dummies.shape[1]})")
        else:
            if not cat.empty:
                messagebox.showinfo("Предупреждение",
                                    "Категориальные столбцы отброшены "
                                    "(снимите галочку, чтобы сохранять только числовые).")
            if num.empty:
                messagebox.showerror("Ошибка", "Не осталось числовых признаков.")
                return
            self._after_load(num, f"Загружено: {fp} (только числовые: {num.shape[1]})")

    def save_csv(self):
        if self.data is None:
            messagebox.showinfo("Информация", "Нет данных для сохранения.")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".csv",
                                          filetypes=[("CSV", "*.csv")])
        if not fp:
            return
        df = self.data.copy()
        if self.labels is not None:
            df["cluster"] = self.labels
        try:
            df.to_csv(fp, index=False)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self._log(f"Сохранено: {fp}")

    # ---------- Feature selection & clustering -----------------------------
    def select_features(self):
        if self.data is None:
            messagebox.showinfo("Инфо", "Сначала загрузите или создайте датасет.")
            return
        k = min(self.k_var.get(), self.data.shape[1]); self.k_var.set(k)
        sub, score = feature_search(self.data, self.radius.get(), k, max_iter=60)
        self.selected, self.labels = sub, None
        names = [self.data.columns[i] for i in sub]
        self._log(f"Выбрано K={k}: {names} | SSE={score:.2f}")
        self._draw_plot(); self.update_dataset_view()

    def cluster(self):
        if self.data is None or self.selected is None:
            messagebox.showinfo("Инфо", "Сначала выберите признаки.")
            return
        X = self.data.iloc[:, self.selected].to_numpy()
        self.labels, _ = forel(X, self.radius.get())
        self._log(f"Кластеров: {len(np.unique(self.labels))} | "
                  f"Общая SSE={cluster_compactness(X, self.labels):.2f}")
        self._draw_plot(); self.update_dataset_view()

    def evaluate_clusters(self):
        if self.labels is None:
            messagebox.showinfo("Инфо", "Сначала кластеризуйте.")
            return
        X = self.data.iloc[:, self.selected].to_numpy()
        for s in cluster_statistics(X, self.labels):
            self._log(f"id={s['cluster']} size={s['size']} "
                      f"sse={s['sse']:.2f} Rmax={s['radius']:.2f}")

    # ---------- Plotting ----------------------------------------------------
    def _draw_plot(self):
        self.ax.clear()
        if self.data is None:
            self.ax.set_title("Нет данных"); self.canvas.draw(); return

        cols = self.selected or list(range(self.data.shape[1]))
        X = self.data.iloc[:, cols].to_numpy()
        if X.shape[1] > 2:
            X2 = PCA(n_components=2).fit_transform(X); xl, yl = "PC1", "PC2"
        else:
            X2 = X[:, :2]; xl = self.data.columns[cols[0]]
            yl = self.data.columns[cols[1]] if len(cols) > 1 else ""

        if self.labels is not None:
            for k in np.unique(self.labels):
                idx = self.labels == k
                self.ax.scatter(X2[idx, 0], X2[idx, 1], label=f"C{k}", s=40, alpha=0.8)
            self.ax.legend(); title = "Кластеры (FOREL)"
        else:
            self.ax.scatter(X2[:, 0], X2[:, 1], color="gray", alpha=0.6, s=30)
            title = "Данные / признаки"

        self.ax.set_title(title); self.ax.set_xlabel(xl); self.ax.set_ylabel(yl)
        self.fig.tight_layout(); self.canvas.draw()

    # ---------- Logging -----------------------------------------------------
    def _log(self, txt: str):
        self.log.insert(tk.END, txt + "\n"); self.log.see(tk.END)


if __name__ == "__main__":
    App().mainloop()
