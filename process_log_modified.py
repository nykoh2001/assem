log_file = open("total_log_m.txt", "w")
for i in range(20):
  with open(f"log_m/{i}.txt", "r") as f:
    lines = f.readlines()
#   with open(f"datas_m/y_test_{i}.txt", "r") as f:
#     y_lines = f.readlines()
#   for l, yl in zip(lines, y_lines):
#     splited_chunks = l.split(" ")
    # origin_length = int(splited_chunks[1])
    # injected_length = len(splited_chunks[6].split(","))
    # output_sequence = yl.split(' ')[4]
    # if injected_length < origin_length:
    #   splited_chunks[0] = "d"
    # splited_chunks[6] = output_sequence
    # log_file.write(" ".join(splited_chunks))
#   f.write(lines)
    for line in lines:
        log_file.write(line)
log_file.close()

import pandas as pd

df = pd.read_csv("total_log_m.txt", delimiter = ' ', names=["타입", "원본시퀀스길이", "오류개수", "실제맞춘개수", "결과종류", "예측시퀀스" ,"정답시퀀스"])
df = df.groupby(['타입','원본시퀀스길이','오류개수', '실제맞춘개수']).count()
# df.columns = ['타입','원본시퀀스길이','오류개수', '실제맞춘개수', "개수"]
df.to_csv("statistics_m.csv")

import json
levels = len(df.index.levels)
dicts = [{} for i in range(levels)]
last_index = None

for index,value, _, _ in df.itertuples():

    if not last_index:
        last_index = index

    for (ii,(i,j)) in enumerate(zip(index, last_index)):
        if not i == j:
            ii = levels - ii -1
            dicts[:ii] =  [{} for _ in dicts[:ii]]
            break

    for i, key in enumerate(reversed(index)):
        dicts[i][key] = value
        value = dicts[i]

    last_index = index

with open("statistics_m.json", 'w') as f:
    json.dump(dicts[-1], f, indent=4)

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

with open("statistics_m.json", "r") as json_file:
    statistics = json.load(json_file)

font_path = "/home/plass-assem/assem/Nayeon/NanumGothic-Regular.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
type_ko = {
    "a":"추가",
    "m":"수정",
    "d":"삭제"
}

# for type in statistics:
exp_file = open("expectations.txt", "w")
current = statistics['m']
for len_sequence in current:
    expectations = []
    x = []
    for num_error in current[len_sequence]:
        x.append(int(num_error))
        expectation = 0
        total_cnt = sum(current[len_sequence][num_error].values())
        for key, cnt in current[len_sequence][num_error].items():
            expectation += int(key) * (int(cnt) / total_cnt)
        expectations.append(expectation)
        
    expectations = list(map(lambda x: (x / int(len_sequence)) * 100, expectations))
    exp_file.write(" ".join(map(str, expectations)) + "\n")
    fig, ax = plt.subplots(figsize=(25, 10))
    plt.rcParams.update({"font.size": 30})
    ax.set_title(f"명령어 시퀀스의 길이 = {len_sequence}", pad=20, fontproperties=fm.FontProperties(fname=font_path))
    ax.set_xticks(x if len(x) <= 10 else list(filter(lambda x: x % 2 == 0, x)))
    ax.set_yticks([12.5*i for i in range(0,9)])
    ax.set_yticklabels(map(lambda x: f"{x}%", [12.5*i for i in range(0,9)]), fontproperties=fm.FontProperties(fname=font_path))
    ax.set_ylim(0, 100)
    ax.set_xlabel("오류가 발생한 명령어 개수", labelpad=10, fontproperties=fm.FontProperties(fname=font_path))
    ax.set_ylabel("평균 일치한 명령어 개수의 비율", labelpad=20,fontproperties=fm.FontProperties(fname=font_path))
    ax.plot(x, expectations)

        # Save the plot
    plot_filename = f"graph_m/{len_sequence}.png"
    plt.savefig(plot_filename)
    plt.clf()


exp_file.close()