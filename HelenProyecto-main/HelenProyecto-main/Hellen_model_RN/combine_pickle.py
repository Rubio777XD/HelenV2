import pickle, os

# Archivos a combinar
# pickle_files = ['data1.pickle', 'data2.pickle', 'data3.pickle','data4.pickle']
pickle_files = [file for file in os.listdir() if file.endswith('.pickle') and file != 'data.pickle']

# Inicializar estructuras combinadas
combined_data = []
combined_labels = []

# Cargar y combinar los datos
for file in pickle_files:
    with open(file, 'rb') as f:
        dataset = pickle.load(f)
        combined_data.extend(dataset['data'])  # Combinar los datos
        combined_labels.extend(dataset['labels'])  # Combinar las etiquetas

# Guardar el dataset combinado
combined_dataset = {'data': combined_data, 'labels': combined_labels}
with open('data.pickle', 'wb') as f:
    pickle.dump(combined_dataset, f)

print(f"Dataset combinado guardado en 'data.pickle'.")
print(f"Número total de muestras: {len(combined_data)}")
print(f"Número total de etiquetas: {len(combined_labels)}")
