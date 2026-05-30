"""Verify notebooks 46, 48, 49, 50 execute without errors.
Mocks matplotlib and sklearn since they're not in the torch env."""
import sys
import types
import importlib.util
import json
import traceback
import glob
import numpy as np


# ── Mock matplotlib ───────────────────────────────────────────────────────────
class MockAx:
    def __getattr__(self, n): return lambda *a, **k: MockAx()
    def __iter__(self): return iter([])
    def __getitem__(self, k): return MockAx()
    def __setitem__(self, k, v): pass
    def table(self, *a, **k):
        t = MockAx()
        t.auto_set_font_size = lambda *a: None
        t.set_fontsize = lambda *a: None
        t.scale = lambda *a: None
        return t

class MockAxArray:
    def __init__(self, nrows, ncols):
        self._data = [[MockAx() for _ in range(ncols)] for _ in range(nrows)]
        self.nrows, self.ncols = nrows, ncols
    def __getitem__(self, k):
        if isinstance(k, tuple): return self._data[k[0]][k[1]]
        row = self._data[k]
        return row[0] if self.ncols == 1 else MockAxRow(row)
    def __iter__(self): return iter(MockAxRow(r) for r in self._data)

class MockAxRow:
    def __init__(self, lst): self._lst = lst
    def __getitem__(self, k): return self._lst[k]
    def __iter__(self): return iter(self._lst)
    def __getattr__(self, n): return lambda *a, **k: MockAx()

class MockFig:
    def __getattr__(self, n): return lambda *a, **k: None

class MockPlt:
    def __getattr__(self, n): return lambda *a, **k: None
    def subplots(self, *a, **k):
        r = a[0] if a else 1
        c = a[1] if len(a) > 1 else 1
        if r == 1 and c == 1: return MockFig(), MockAx()
        if r == 1: return MockFig(), MockAxRow([MockAx() for _ in range(c)])
        return MockFig(), MockAxArray(r, c)
    Rectangle = staticmethod(lambda *a, **k: MockAx())
    colorbar = staticmethod(lambda *a, **k: MockAx())
    show = tight_layout = savefig = staticmethod(lambda *a, **k: None)

mock_plt = MockPlt()
mat = types.ModuleType('matplotlib')
mat.pyplot = mock_plt
sys.modules['matplotlib'] = mat
sys.modules['matplotlib.pyplot'] = mock_plt


# ── Mock sklearn ──────────────────────────────────────────────────────────────
class LinReg:
    def fit(self, X, y):
        X_ = np.array(X)
        if X_.ndim == 1:
            X_ = X_.reshape(-1, 1)
        Xb = np.column_stack([X_, np.ones(len(X_))])
        w = np.linalg.lstsq(Xb, np.array(y), rcond=None)[0]
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        return self

    def predict(self, X):
        X_ = np.array(X)
        if X_.ndim == 1:
            X_ = X_.reshape(-1, 1)
        return X_ @ self.coef_ + self.intercept_


class Ridge(LinReg):
    def __init__(self, alpha=1.0, **k):
        self.alpha = alpha


class GBR:
    def __init__(self, **k):
        self.feature_importances_ = None

    def fit(self, X, y):
        X_ = np.array(X)
        self._lr = LinReg().fit(X_, np.array(y))
        self.feature_importances_ = np.ones(X_.shape[1]) / X_.shape[1]
        return self

    def predict(self, X): return self._lr.predict(X)


class GBC:
    def __init__(self, **k):
        self.feature_importances_ = None

    def fit(self, X, y):
        X_ = np.array(X)
        y_ = np.array(y).astype(float)
        self._lr = LinReg().fit(X_, y_)
        self.feature_importances_ = np.ones(X_.shape[1]) / X_.shape[1]
        return self

    def predict_proba(self, X):
        p = np.clip(self._lr.predict(X), 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class LogReg:
    def __init__(self, **k): pass

    def fit(self, X, y):
        X_ = np.array(X)
        self._lr = LinReg().fit(X_, np.array(y).astype(float))
        return self

    def predict_proba(self, X):
        p = np.clip(self._lr.predict(X), 0, 1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StdScaler:
    def fit(self, X):
        X_ = np.array(X)
        self.mean_ = X_.mean(0)
        self.scale_ = X_.std(0) + 1e-9
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (np.array(X) - self.mean_) / self.scale_


def mae(y_true, y_pred, **k):
    return float(np.abs(np.array(y_true) - np.array(y_pred)).mean())


def r2(y_true, y_pred):
    y = np.array(y_true)
    p = np.array(y_pred)
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2 + 1e-9).sum()


def precision_score(y, p, **k):
    y, p = np.array(y), np.array(p)
    return float((y[p == 1] == 1).mean()) if (p == 1).any() else 0.0


def recall_score(y, p, **k):
    y, p = np.array(y), np.array(p)
    return float((p[y == 1] == 1).mean()) if (y == 1).any() else 0.0


def f1_score(y, p, **k):
    pr = precision_score(y, p)
    re = recall_score(y, p)
    return 2 * pr * re / (pr + re + 1e-9)


def roc_auc_score(y, p):
    y, p = np.array(y), np.array(p)
    order = np.argsort(-p)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / (tps[-1] + 1e-9)
    fpr = fps / (fps[-1] + 1e-9)
    # np.trapz renamed to np.trapezoid in numpy 2.x
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
    return float(trapz(tpr, fpr))


def roc_curve(y, p):
    fpr_list, tpr_list = [0.0], [0.0]
    for t in np.linspace(1, 0, 20):
        pred = (p >= t).astype(int)
        fpr_list.append(1 - precision_score(1 - y, 1 - pred))
        tpr_list.append(recall_score(y, pred))
    return np.array(fpr_list), np.array(tpr_list), np.linspace(1, 0, 21)


def classification_report(y, p, **k):
    return f'accuracy={np.mean(np.array(y) == np.array(p)):.3f}\n'


def _make_mod(name, **attrs):
    m = types.ModuleType(name)
    # __spec__ required by torch._dynamo.trace_rules scanning sys.modules
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    m.__path__ = []  # mark as package so sub-imports work
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

sk_linear = _make_mod('sklearn.linear_model',
    LinearRegression=LinReg, Ridge=Ridge, LogisticRegression=LogReg)
sk_ensemble = _make_mod('sklearn.ensemble',
    RandomForestRegressor=GBR, GradientBoostingRegressor=GBR,
    GradientBoostingClassifier=GBC)
sk_metrics = _make_mod('sklearn.metrics',
    mean_absolute_error=mae, r2_score=r2,
    precision_score=precision_score, recall_score=recall_score,
    f1_score=f1_score, roc_auc_score=roc_auc_score,
    roc_curve=roc_curve, classification_report=classification_report)
sk_preprocessing = _make_mod('sklearn.preprocessing', StandardScaler=StdScaler)

sk = _make_mod('sklearn',
    linear_model=sk_linear, ensemble=sk_ensemble,
    metrics=sk_metrics, preprocessing=sk_preprocessing)

sys.modules['sklearn'] = sk
sys.modules['sklearn.linear_model'] = sk_linear
sys.modules['sklearn.ensemble'] = sk_ensemble
sys.modules['sklearn.metrics'] = sk_metrics
sys.modules['sklearn.preprocessing'] = sk_preprocessing


# ── Run notebooks ─────────────────────────────────────────────────────────────
results = {}
for n in ['46', '48', '49', '50']:
    fs = glob.glob(f'modern-ai/notebooks/{n}-*.ipynb')
    if not fs:
        print(f'{n}: NOT FOUND')
        results[n] = False
        continue
    f = fs[0]
    name = f.split('/')[-1]
    print(f'\n=== {name} ===')
    nb = json.load(open(f))
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    ns = {}
    all_ok = True
    for i, cell in enumerate(code_cells):
        src = ''.join(cell['source'])
        try:
            exec(src, ns)
            print(f'  Cell {i + 1}: OK')
        except Exception as e:
            print(f'  Cell {i + 1}: ERROR {type(e).__name__}: {e}')
            for ln in traceback.format_exc().split('\n')[-8:]:
                if ln.strip():
                    print(f'    {ln}')
            all_ok = False
            break
    results[n] = all_ok
    print(f'{"PASS" if all_ok else "FAIL"}: {name}')

print('\n=== SUMMARY ===')
for n, ok in results.items():
    print(f'  {n}: {"PASS" if ok else "FAIL"}')
all_pass = all(results.values())
print(f'\nAll pass: {all_pass}')
sys.exit(0 if all_pass else 1)
