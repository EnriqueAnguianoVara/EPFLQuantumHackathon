# 🧪 Guía de Setup y Verificación

## 1. Instalación (una sola línea)

```bash
pip install numpy pandas scikit-learn openpyxl torch plotly streamlit xgboost lightgbm joblib matplotlib merlinquantum perceval-quandela
```

> **Nota:** Si estás en un entorno con restricciones (Colab, sistema con python gestionado), añade `--break-system-packages` al final.

> **Nota 2:** Si `merlinquantum` o `perceval-quandela` fallan, instala primero el resto y luego:
> ```bash
> pip install merlinquantum perceval-quandela
> ```
> MerLin requiere Python ≥3.9 y < 3.13.

---

## 2. Verificación rápida (sin Streamlit)

Desde la raíz del proyecto (`quantum-swaptions/`), ejecuta:

```bash
python test_phase1.py
```

Este script verifica todas las piezas de la Fase 1 sin necesitar plotly ni streamlit.

---

## 3. Verificación completa con Streamlit

```bash
cd quantum-swaptions
streamlit run app.py
```

Se abre en `http://localhost:8501`. Navega a **📊 Market Explorer** y verifica:
- El slider de fecha mueve el heatmap
- Las series temporales se dibujan al seleccionar tenors
- El PCA muestra que 3 componentes > 99%
- El test template muestra 6 future + 2 missing rows

---

## 4. Qué verificar en cada módulo

### `src/data/loader.py`
```python
from src.data.loader import load_all
data = load_all()
assert data['train_prices'].shape == (494, 224)
assert len(data['test_info']['future_indices']) == 6
assert len(data['test_info']['missing_indices']) == 2
print("✓ loader OK")
```

### `src/data/preprocessing.py`
```python
from src.data.preprocessing import full_pipeline
pipe = full_pipeline(data['train_prices'], n_pca_components=6, window_size=20)
assert pipe['X_train'].shape[1] == 20 * 6  # window_size * n_pca
assert pipe['Y_train'].shape[1] == 6       # n_pca
print("✓ preprocessing OK")
```

### `src/utils/surface.py`
```python
from src.utils.surface import flat_to_grid, grid_to_flat
import numpy as np
flat = data['train_prices'][0]
grid = flat_to_grid(flat)
assert grid.shape == (14, 16)
assert np.allclose(flat, grid_to_flat(grid))
print("✓ surface OK")
```

### `src/evaluation/metrics.py`
```python
from src.evaluation.metrics import all_metrics
import numpy as np
y = np.random.randn(10, 224)
m = all_metrics(y, y + 0.01 * np.random.randn(10, 224))
assert 'MAE' in m and 'RMSE' in m and 'R²' in m
print("✓ metrics OK")
```

---

## 5. Problemas comunes

| Problema | Solución |
|----------|----------|
| `ModuleNotFoundError: No module named 'src'` | Ejecuta desde la raíz del proyecto `quantum-swaptions/`, o añade `sys.path.insert(0, '/ruta/al/quantum-swaptions')` |
| `plotly` no se instala | `pip install plotly==5.18.0` (versión específica) |
| `merlinquantum` falla al instalar | Verifica Python 3.9-3.12. En Mac M1/M2: `pip install --no-binary :all: merlinquantum` |
| Streamlit no abre | Verifica que el puerto 8501 esté libre. Usa `streamlit run app.py --server.port 8502` |
| `FileNotFoundError: train.xlsx` | Asegúrate de que `data/train.xlsx` existe en el directorio del proyecto |
| Las fechas se parsean mal | El loader asume formato `dd/mm/yyyy` (dayfirst=True). Si tu Excel guarda otro formato, ajusta en `loader.py` |
