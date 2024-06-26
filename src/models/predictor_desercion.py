import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import json

# Cargar y preprocesar los datos de entrenamiento
def cargar_datos_entrenamiento(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    
    # Codificar variables categóricas
    le = LabelEncoder()
    df['genero'] = le.fit_transform(df['genero'])
    df['situacion_familiar'] = le.fit_transform(df['situacion_familiar'])
    
    # Crear variable objetivo (1 si desertó, 0 en otro caso)
    df['desercion'] = (df['estado'] == 'Desertó').astype(int)
    
    return df

# Entrenar el modelo con los datos de entrenamiento
def entrenar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    return modelo, X_test, y_test

# Evaluar el modelo entrenado
def evaluar_modelo(modelo, X_test, y_test):
    predicciones = modelo.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)
    
    return precision, reporte

# Cargar datos de nuevos estudiantes para predicción
def cargar_datos_nuevos(ruta_archivo):
    df = pd.read_csv(ruta_archivo)
    
    # Codificar variables categóricas
    le = LabelEncoder()
    df['genero'] = le.fit_transform(df['genero'])
    df['situacion_familiar'] = le.fit_transform(df['situacion_familiar'])
    
    return df

# Predecir el riesgo de deserción para nuevos estudiantes
def predecir_nuevos_estudiantes(modelo, df_nuevos):
    caracteristicas = ['edad', 'genero', 'promedio_academico', 'asistencia', 'estrato_socioeconomico', 
                       'trabaja', 'actividades_extracurriculares', 'situacion_familiar', 'semestre']
    
    df_nuevos['riesgo_desercion'] = modelo.predict_proba(df_nuevos[caracteristicas])[:, 1]
    
    return df_nuevos

def main():
    # Entrenar el modelo con los datos de entrenamiento
    df_entrenamiento = cargar_datos_entrenamiento('perfiles_estudiantes.csv')
    
    caracteristicas = ['edad', 'genero', 'promedio_academico', 'asistencia', 'estrato_socioeconomico', 
                       'trabaja', 'actividades_extracurriculares', 'situacion_familiar', 'semestre']
    
    X = df_entrenamiento[caracteristicas]
    y = df_entrenamiento['desercion']
    
    modelo, X_test, y_test = entrenar_modelo(X, y)
    
    precision, reporte = evaluar_modelo(modelo, X_test, y_test)
    print(f"Precisión del modelo: {precision}")
    print("Reporte de clasificación:")
    print(reporte)
    
    # Cargar nuevos datos para predicción
    df_nuevos = cargar_datos_nuevos('nuevos_estudiantes.csv')
    
    # Generar predicciones para los nuevos estudiantes
    resultados_nuevos = predecir_nuevos_estudiantes(modelo, df_nuevos)
    
    # Guardar resultados en JSON
    resultados_json = resultados_nuevos[['id', 'riesgo_desercion']].to_dict('records')
    with open('resultados_nuevos.json', 'w') as f:
        json.dump(resultados_json, f)

if __name__ == "__main__":
    main()
