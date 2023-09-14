import sys
import pickle

process_num = int(sys.argv[1])

with open("X_test_m_total.pickle","rb") as f:
    X_test_total = pickle.load(f)
with open("y_test_m_total.pickle","rb") as f:
    y_test_total = pickle.load(f)

dataset_size = 200000 // process_num

print(X_test_total[0])

for i in range(0, process_num):
  X_file = open(f"datas_m/X_test_{i}.txt", "w")
  y_file = open(f"datas_m/y_test_{i}.txt", "w")
  for j in range(dataset_size * i, dataset_size * (i+1)):
    X_file.write(X_test_total[j] + "\n")
    y_file.write(y_test_total[j] + "\n")
  X_file.close()
  y_file.close()
