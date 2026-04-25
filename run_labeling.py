from labeling import data_labeling, dataset_building
label = input("Enter label(figure, gesture): ")
count = int(input("How many samples: "))

data_labeling(count, label)
dataset_building("data/dataset.pkl")