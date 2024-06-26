import pandas as pd
import numpy as np

# Función para generar perfiles de estudiantes
def generar_perfiles(n=1000):
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
    
    # Generar el estado del estudiante basado en algunas reglas simples
    def determinar_estado(row):
        if row['promedio_academico'] < 3.0 or row['asistencia'] < 70:
            return 'Desertó'
        elif row['semestre'] == 10 and row['promedio_academico'] >= 3.5:
            return 'Graduado'
        else:
            return 'Estudiante'
    
    df['estado'] = df.apply(determinar_estado, axis=1)
    
    return df

# Generar perfiles y guardar en CSV
perfiles = generar_perfiles(1000)
perfiles.to_csv('perfiles_estudiantes.csv', index=False)
print(perfiles.head())
print(perfiles['estado'].value_counts(normalize=True))