# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dashboard de Detección de Intrusiones",
    page_icon="🛡️",
    layout="wide"
)

# --- Funciones de Utilidad para Gráficos ---
def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """
    Crea una matriz de confusión interactiva con Plotly.
    `class_names` debe ser una lista con los nombres de las clases en el orden correcto (ej: ['Normal', 'Ataque'])
    """
    # 1. Usar las etiquetas numéricas (0, 1, etc.) para el cálculo
    numeric_labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=numeric_labels)
    
    # 2. Crear el gráfico usando los nombres de las clases para los ejes
    fig = px.imshow(cm,
                    labels=dict(x="Predicción", y="Valor Real", color="Frecuencia"),
                    x=class_names, # Usar los nombres de las clases aquí
                    y=class_names, # y aquí
                    text_auto=True,
                    color_continuous_scale='Blues')
    
    fig.update_layout(title_text=title, title_x=0.5)
    return fig

def plot_feature_importance(model, columns, title):
    """Crea un gráfico de barras interactivo para la importancia de características."""
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'feature': columns, 'importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(15)
    fig = px.bar(feat_imp_df,
                 x='importance',
                 y='feature',
                 orientation='h',
                 title=title)
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig


# --- Funciones con Caché para Mejorar el Rendimiento ---
@st.cache_data
def cargar_datos(filepath):
    """Carga el conjunto de datos desde un archivo CSV."""
    df = pd.read_csv(filepath)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df

@st.cache_data
def preprocesar_para_binario(df):
    """Preprocesa los datos para la clasificación binaria."""
    df_cleaned = df.copy()
    for col in ['dur', 'rate']:
        subset = df_cleaned[df_cleaned['label'] == 0][col]
        q1 = subset.quantile(0.25)
        q3 = subset.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        df_cleaned = df_cleaned[~((df_cleaned['label'] == 0) & (df_cleaned[col] > upper_bound))]
    columns_to_encode = ['proto', 'service', 'state']
    for col in columns_to_encode:
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])
    if 'attack_cat' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['attack_cat'])
    X = df_cleaned.drop('label', axis=1)
    y = df_cleaned['label'].astype(int)
    return X, y

@st.cache_resource
def entrenar_modelos_binarios(X, y):
    """Entrena y devuelve los modelos de clasificación binaria y el escalador."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=10)
    xgb_clf.fit(X_train_scaled, y_train)
    models = {'RandomForest': rf_clf, 'XGBoost': xgb_clf}
    return models, scaler, X_test_scaled, y_test

@st.cache_data
def preprocesar_para_multiclase(df):
    """Preprocesa los datos para la clasificación multiclase."""
    df_attacks = df[df['label'] == 1].copy()
    attack_counts = df_attacks['attack_cat'].value_counts()
    minority_categories = attack_counts[attack_counts < 5000].index
    df_attacks['attack_cat_grouped'] = df_attacks['attack_cat'].apply(
        lambda x: 'Otros' if x in minority_categories else x
    )
    features_to_use = [col for col in df.columns if col not in ['label', 'attack_cat', 'attack_cat_grouped']]
    X_multiclass = df_attacks[features_to_use].copy()
    y_multiclass = df_attacks['attack_cat_grouped']
    categorical_cols = ['proto', 'service', 'state']
    for col in categorical_cols:
        le = LabelEncoder()
        X_multiclass[col] = le.fit_transform(X_multiclass[col].astype(str))
    target_encoder = LabelEncoder()
    y_multiclass_encoded = target_encoder.fit_transform(y_multiclass)
    return X_multiclass, y_multiclass_encoded, target_encoder

@st.cache_resource
def entrenar_modelos_multiclase(X, y, _target_encoder):
    """Entrena los modelos multiclase."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_multi = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_multi.fit(X_train_scaled, y_train)
    xgb_multi = XGBClassifier(
        objective='multi:softprob', n_estimators=200, max_depth=10, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    xgb_multi.fit(X_train_scaled, y_train)
    models = {'RandomForest': rf_multi, 'XGBoost': xgb_multi}
    return models, scaler, X_test_scaled, y_test

# --- Aplicación Principal ---
st.title("🛡️ Dashboard de Detección de Intrusiones de Red")

# --- Barra Lateral de Navegación ---
st.sidebar.header("Navegación")
page = st.sidebar.radio("Ir a", [
    "1. Introducción y Vista General",
    "2. Análisis Exploratorio de Datos (AED)",
    "3. Clasificación Binaria (Ataque vs. Normal)",
    "4. Clasificación Multiclase (Tipos de Ataque)",
    "5. Predicción en Vivo"
])

try:
    df_raw = cargar_datos("UNSW_NB15_testing-set.csv")
except FileNotFoundError:
    st.error("Error: No se encontró `UNSW_NB15_testing-set.csv`. Por favor, coloca el archivo en el mismo directorio que `app.py`.")
    st.stop()

# --- Página 1: Introducción y Vista General ---
if page == "1. Introducción y Vista General":
    st.header("Resumen del Proyecto")
    st.markdown("""
    Este dashboard proporciona un análisis completo del conjunto de datos **UNSW-NB15**. El objetivo es construir modelos de aprendizaje automático capaces de distinguir entre el tráfico de red normal y los ataques maliciosos.
    El proyecto incluye dos tareas de modelado:
    1.  **Clasificación Binaria**: Identificar si una conexión es `Normal` o un `Ataque`.
    2.  **Clasificación Multiclase**: Si es un ataque, identificar el `tipo de ataque` específico.
    """)
    st.header("Muestra de Datos e Información Básica")
    st.dataframe(df_raw.head())
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dimensiones de los Datos")
        st.write(f"Número de Filas: {df_raw.shape[0]}")
        st.write(f"Número de Columnas: {df_raw.shape[1]}")
    with col2:
        st.subheader("Tipos de Datos")
        st.dataframe(df_raw.dtypes.astype(str))
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df_raw.describe())

# --- Página 2: Análisis Exploratorio de Datos (AED) ---
elif page == "2. Análisis Exploratorio de Datos (AED)":
    st.header("Análisis Exploratorio de Datos")
    #st.info("Estos gráficos son interactivos. Puedes hacer zoom, desplazarte y ver detalles pasando el cursor por encima.")

    st.subheader("Distribución del Tráfico: Normal vs. Ataque")
    label_counts = df_raw['label'].value_counts().reset_index()
    label_counts.columns = ['label', 'count']
    label_counts['label'] = label_counts['label'].map({0: 'Normal', 1: 'Ataque'})
    fig_labels = px.bar(label_counts, x='label', y='count',
                        title="Distribución de Etiquetas (Normal vs. Ataque)",
                        color='label', color_discrete_map={'Normal':'#1f77b4', 'Ataque':'#d62728'},
                        labels={'label':'Tipo de Tráfico', 'count':'Frecuencia'})
    st.plotly_chart(fig_labels, use_container_width=True)

    st.subheader("Distribución de Categorías de Ataque")
    attack_cat_counts = df_raw[df_raw['label'] == 1]['attack_cat'].value_counts().reset_index()
    attack_cat_counts.columns = ['category', 'count']
    fig_attacks = px.bar(attack_cat_counts, x='category', y='count',
                         title="Frecuencia de Cada Categoría de Ataque",
                         labels={'category': 'Categoría de Ataque', 'count': 'Frecuencia'})
    fig_attacks.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_attacks, use_container_width=True)

    st.subheader("Diagramas de Caja Interactivos por Etiqueta")
    st.markdown("El eje Y está en escala logarítmica para una mejor visualización de los valores extremos.")
    features_to_plot = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate']
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features_to_plot):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            fig = px.box(df_raw, x='label', y=feature,
                         title=f'Distribución de {feature}',
                         color='label',
                         labels={'label': 'Tipo de Tráfico (0: Normal, 1: Ataque)'})
            fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)

# --- Página 3: Clasificación Binaria ---
elif page == "3. Clasificación Binaria (Ataque vs. Normal)":
    st.header("Clasificación Binaria: Ataque vs. Normal")
    st.markdown("Modelos para clasificar el tráfico como 'Normal' (0) o 'Ataque' (1).")
    with st.spinner("Preprocesando datos y entrenando modelos..."):
        X_binary, y_binary = preprocesar_para_binario(df_raw)
        binary_models, _, X_test_scaled, y_test = entrenar_modelos_binarios(X_binary, y_binary)
    st.success("¡Modelos entrenados con éxito!")

    model_choice = st.selectbox("Elige un modelo para evaluar:", ("RandomForest", "XGBoost"))
    model = binary_models[model_choice]

    st.subheader(f"Métricas de Evaluación para {model_choice}")
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Normal', 'Ataque'])
    st.dataframe(pd.DataFrame(report).transpose())

    col1, col2 = st.columns(2)
    with col1:
        fig_cm = plot_confusion_matrix(y_test, y_pred, class_names=['Normal', 'Ataque'], title="Matriz de Confusión")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        fig_fi = plot_feature_importance(model, X_binary.columns, f"Top 15 Características ({model_choice})")
        st.plotly_chart(fig_fi, use_container_width=True)

# --- Página 4: Clasificación Multiclase ---
elif page == "4. Clasificación Multiclase (Tipos de Ataque)":
    st.header("Clasificación Multiclase: Identificando Tipos de Ataque")
    st.markdown("Dado que una conexión es un ataque, este modelo identifica el tipo específico.")
    with st.spinner("Preprocesando y entrenando modelos multiclase..."):
        X_multi, y_multi_encoded, target_encoder = preprocesar_para_multiclase(df_raw)
        multi_models, _, X_test_multi_scaled, y_test_multi_encoded = entrenar_modelos_multiclase(X_multi, y_multi_encoded, target_encoder)
    st.success("¡Modelos multiclase entrenados con éxito!")

    model_choice_multi = st.selectbox("Elige un modelo multiclase para evaluar:", ("RandomForest", "XGBoost"))
    model_multi = multi_models[model_choice_multi]
    y_pred_multi_encoded = model_multi.predict(X_test_multi_scaled)

    y_test_multi_decoded = target_encoder.inverse_transform(y_test_multi_encoded)
    y_pred_multi_decoded = target_encoder.inverse_transform(y_pred_multi_encoded)

    st.subheader(f"Métricas de Evaluación para {model_choice_multi}")
    # El reporte de clasificación SÍ necesita los datos decodificados, así que esto está bien.
    report_multi = classification_report(y_test_multi_decoded, y_pred_multi_decoded, output_dict=True)
    st.dataframe(pd.DataFrame(report_multi).transpose())

    col1, col2 = st.columns(2)
    with col1:
        # CORRECCIÓN: Usar los datos codificados (numéricos) para la matriz de confusión.
        # La función se encargará de poner los nombres de las clases en los ejes.
        fig_cm_multi = plot_confusion_matrix(y_test_multi_encoded, y_pred_multi_encoded, class_names=target_encoder.classes_, title="Matriz de Confusión Multiclase")
        st.plotly_chart(fig_cm_multi, use_container_width=True)
    with col2:
        fig_fi_multi = plot_feature_importance(model_multi, X_multi.columns, f"Top 15 Características ({model_choice_multi})")
        st.plotly_chart(fig_fi_multi, use_container_width=True)

# --- Página 5: Predicción en Vivo ---
elif page == "5. Predicción en Vivo":
    st.header("Predicción de Tráfico de Red en Vivo")
    st.markdown("Introduce las características de una conexión para obtener una predicción.")

    with st.spinner("Cargando modelos y preprocesadores..."):
        X_binary, y_binary = preprocesar_para_binario(df_raw)
        binary_models, binary_scaler, _, _ = entrenar_modelos_binarios(X_binary, y_binary)
        X_multi, y_multi_encoded, target_encoder = preprocesar_para_multiclase(df_raw)
        multi_models, multi_scaler, _, _ = entrenar_modelos_multiclase(X_multi, y_multi_encoded, target_encoder)

    st.subheader("Introduce las Características de la Conexión:")
    input_data = {}
    cols = st.columns(3)
    df_for_options = cargar_datos("UNSW_NB15_testing-set.csv")
    
    with cols[0]:
        input_data['proto'] = st.selectbox("Protocolo (proto)", options=df_for_options['proto'].unique())
        input_data['service'] = st.selectbox("Servicio (service)", options=df_for_options['service'].unique())
        input_data['state'] = st.selectbox("Estado (state)", options=df_for_options['state'].unique())

    # Características numéricas importantes para la entrada del usuario
    imp_features = ['sttl', 'dttl', 'sbytes', 'dbytes', 'rate', 'sload', 'dload',
                    'spkts', 'dpkts', 'dur', 'smean', 'dmean', 'ct_dst_src_ltm',
                    'ct_state_ttl', 'ct_srv_src']
    
    current_col_idx = 1
    for i, feature in enumerate(imp_features):
        with cols[current_col_idx]:
            input_data[feature] = st.number_input(f"{feature}", value=float(df_raw[feature].mean()), format="%.4f")
        if (i + 1) % 5 == 0: # Cambiar de columna cada 5 características
             current_col_idx = (current_col_idx % 2) + 1


    full_feature_list = X_binary.columns
    for feature in full_feature_list:
        if feature not in input_data:
            input_data[feature] = df_raw[feature].mean()

    if st.button("Predecir", type="primary"):
        input_df = pd.DataFrame([input_data])
        input_df_processed = input_df.copy()
        for col in ['proto', 'service', 'state']:
            le = LabelEncoder()
            le.fit(df_for_options[col].astype(str))
            input_df_processed[col] = le.transform(input_df_processed[col].astype(str))
        input_df_processed = input_df_processed[X_binary.columns]
        input_scaled = binary_scaler.transform(input_df_processed)

        st.subheader("Resultados de la Predicción")
        binary_model = binary_models['XGBoost']
        binary_pred = binary_model.predict(input_scaled)[0]
        binary_proba = binary_model.predict_proba(input_scaled)[0]

        if binary_pred == 1:
            st.error(f"**Predicción: ATAQUE** (Confianza: {binary_proba[1]:.2%})", icon="🚨")
            multi_model = multi_models['XGBoost']
            input_multi_processed = input_df_processed[X_multi.columns]
            input_multi_scaled = multi_scaler.transform(input_multi_processed)
            multi_pred_encoded = multi_model.predict(input_multi_scaled)[0]
            multi_pred_decoded = target_encoder.inverse_transform([multi_pred_encoded])[0]
            multi_proba = multi_model.predict_proba(input_multi_scaled)[0]
            st.warning(f"**Tipo de Ataque Predicho: {multi_pred_decoded}** (Confianza: {np.max(multi_proba):.2%})", icon="🔍")
        else:
            st.success(f"**Predicción: NORMAL** (Confianza: {binary_proba[0]:.2%})", icon="✅")