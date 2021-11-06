from src.si.data.dataset import Dataset, summary
dataseteste = Dataset.from_data("C:/Users/Ze/Desktop/Mestrado/3ºSemestre/si/datasets/breast-bin.data", labeled=False)
print(dataseteste.Y)
print(dataseteste.X)
# teste
print(dataseteste.X - 1)

print(summary(dataseteste))

from src.si.util.scale import StandardScaler
scaler = StandardScaler()
scaler.fit(dataseteste)
scaler.transform(dataseteste)
datascaled = scaler.fit_transform(dataseteste)

