import pandas as pd
import numpy as np

# Función para generar nuevos perfiles de estudiantes sin la columna 'estado'
def generar_nuevos_perfiles(n=1000):
    np.random.seed(42)  # Para reproducibilidad
    
    data = {
        'id': range(1, n+1),
        'edad': np.random.randint(18, 30, n),
        'genero': np.random.choice(['M', 'F'], n),
        'promedio_academico': np.round(np.random.uniform(2.0, 5.0, n), 2),
        'asistencia': np.random.randint(60, 100, n),
        'estrato_socioeconomico': np.random.randint(1, 7, n),
        'trabaja': np.random.choice([0, 1], n),
        'actividades_extracurriculares': np.random.randint(0, 5, n),
        'situacion_familiar': np.random.choice(['Estable', 'Complicada'], n),
        'semestre': np.random.randint(1, 11, n)
    }
    
    df = pd.DataFrame(data)
    
    return df

# Generar nuevos perfiles y guardar en CSV
nuevos_perfiles = generar_nuevos_perfiles(1000)
nuevos_perfiles.to_csv('nuevos_estudiantes.csv', index=False)
print(nuevos_perfiles.head())
