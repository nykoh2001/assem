# sorted_log = open("sorted_log_m.txt", 'w')
# with open("total_log_m.txt") as log_file:
#   lines = log_file.readlines()
# lines = sorted(lines)
# for line in lines:
#   sorted_log.write(line)

# sorted_log.close()

sorted_log = open("sorted_log_m.txt", 'w')
with open("total_log_m.txt", 'r') as f :
  lines = f.readlines()
  lines = sorted(lines)
  for line in lines:
    sorted_log.write(line)
sorted_log.close()

sorted_log = open("sorted_log_m.txt", 'r')
lines = sorted_log.readlines()
lines = [line.split(" ") for line in lines]

curr_idx = 0
prev = lines[curr_idx]
current = lines[curr_idx]

while curr_idx < 6000:
  for length, err in [tuple(current[1:3])]:
    sep_log = open(f"sep_log/{length}_{err}.txt", "w")
    sep_log.write(" ".join(current))
    curr_idx += 1
    current = lines[curr_idx]
    prev = lines[curr_idx - 1]
    while prev[1] == current[1] and prev[2] == current[2]:
      sep_log.write(" ".join(current))
      curr_idx += 1
      if curr_idx == 6000:
        break
      current = lines[curr_idx]
      prev = lines[curr_idx - 1]
    sep_log.close()
sorted_log.close()