from nLCS import nLCS
from levi import similarity_with_levenshtein_distance
from accuracy import get_accuracy
from hamming import hamming

import statistics

levi_file = open("levi.txt", "w")
nLSC_file = open("nLCS.txt", "w")
acc_file = open("accuracy.txt", "w")
hamming_file = open("hamming.txt", "w")


for i in range(5, 41, 5):
    percentage_thres = i * 0.4
    for j in range(int(percentage_thres) + 1):
        file = open(f"sep_log/{i}_{j}.txt", "r")
        lines = file.readlines()
        
        levi_sim = []
        nLCS_sim = []
        acc = []
        hamm = []
        for line in lines:
          chunks = line.rstrip().split(" ")
          len_cor = chunks[1:4]
          x_y = chunks[5:7]
          
          levi_sim.append(similarity_with_levenshtein_distance(x_y[0], x_y[1]))
          nLCS_sim.append(nLCS(x_y[0], x_y[1]))
          acc.append(get_accuracy(len_cor[0], len_cor[2]))
          hamm.append(hamming(len_cor[0], len_cor[2]))
        print(str(statistics.mean(acc)))
        levi_file.write(f"{i} {j} {str(statistics.mean(levi_sim))}\n")
        nLSC_file.write(f"{i} {j} {str(statistics.mean(nLCS_sim))}\n")
        acc_file.write(f"{i} {j} {str(statistics.mean(acc))}\n")
        hamming_file.write(f"{i} {j} {str(statistics.mean(hamm))}\n")
        file.close()
levi_file.close()
nLSC_file.close()
acc_file.close()
hamming_file.close()