# Dashboard de Detección de Intrusiones de Red

Este es un dashboard interactivo construido con Streamlit para analizar el conjunto de datos **UNSW-NB15** y entrenar modelos de Machine Learning para la detección de intrusiones de red.

La aplicación permite explorar los datos, visualizar el rendimiento de diferentes modelos de clasificación y realizar predicciones en vivo sobre nuevas conexiones de red.

  <!-- Opcional: Sube un screenshot a imgur.com y pega el enlace aquí -->

---

## Características

-   **Análisis Exploratorio de Datos (AED)**: Visualizaciones interactivas de la distribución de datos, tipos de ataque y características clave.
-   **Clasificación Binaria**: Modelos (Random Forest y XGBoost) para clasificar el tráfico como `Normal` o `Ataque`.
-   **Clasificación Multiclase**: Modelos para identificar el tipo específico de ataque (`Generic`, `Exploits`, `Fuzzers`, etc.).
-   **Evaluación de Modelos**: Métricas detalladas, matrices de confusión interactivas e importancia de características para cada modelo.
-   **Predicción en Vivo**: Una interfaz para introducir datos de una conexión de red y obtener una predicción instantánea del modelo.

---

## Tecnologías Utilizadas

-   **Python**: Lenguaje de programación principal.
-   **Streamlit**: Framework para construir la aplicación web interactiva.
-   **Pandas**: Para la manipulación y análisis de datos.
-   **Scikit-learn**: Para el preprocesamiento de datos y la creación de modelos de Machine Learning.
-   **XGBoost**: Para entrenar un modelo de Gradient Boosting de alto rendimiento.
-   **Plotly**: Para crear visualizaciones de datos interactivas y modernas.

---

## Instalación y Ejecución Local

Sigue estos pasos para ejecutar el dashboard en tu máquina local.

### Prerrequisitos

-   Python 3.9+
-   Git

### 1. Clonar el Repositorio

```bash
git clone https://github.com/felipearosr/intrusion-detection-dashboard
cd intrusion-detection-dashboard
```


### 2. Crear un Entorno Virtual (Recomendado)

```bash
# Crear el entorno
python -m venv venv

# Activar el entorno
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar las Dependencias

Todas las librerías necesarias están listadas en `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Descargar el Conjunto de Datos

Este proyecto utiliza el archivo `UNSW_NB15_testing-set.csv`. Debido a su tamaño, no está incluido en este repositorio. Por favor, descarga el archivo desde la fuente original (o tu fuente) y colócalo en la raíz del proyecto.

> **Fuente del Dataset:** Puedes encontrar el conjunto de datos UNSW-NB15 en [este enlace de Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) o en el sitio web de la Universidad de Nueva Gales del Sur.

La estructura de tu directorio debe ser:
```
mi-dashboard-deteccion-intrusiones/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── UNSW_NB15_testing-set.csv  <-- ¡Coloca el archivo aquí!
```

### 5. Ejecutar la Aplicación

Una vez que todo esté configurado, inicia la aplicación Streamlit:

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.
