import sys
import pickle

process_num = int(sys.argv[1])

with open("Nayeon/X_modified_total_40_test.pickle","rb") as f:
    X_test_total = pickle.load(f)
with open("Nayeon/y_modified_total_40_test.pickle","rb") as f:
    y_test_total = pickle.load(f)
with open("Nayeon/metadata_modified_total_40_test.pickle","rb") as f:
    metadata_test_total = pickle.load(f)

print(len(X_test_total))
print(len(y_test_total))
print(len(metadata_test_total))

dataset_size = 200000 // process_num

print(X_test_total[0])
print(y_test_total[0])
print(metadata_test_total[0])

for i in range(0, process_num):
  X_file = open(f"Nayeon/datas_m/X_test_{i}.txt", "w")
  y_file = open(f"Nayeon/datas_m/y_test_{i}.txt", "w")
  meta_file=open(f"Nayeon/datas_m/meta_test_{i}.txt", "w")
   
  for j in range(dataset_size * i, dataset_size * (i+1)):
    X_file.write(X_test_total[j] + "\n")
    y_file.write(y_test_total[j] + "\n")
    meta_file.write(" ".join(metadata_test_total[j]) + "\n")
  X_file.close()
  y_file.close()
  meta_file.close()
