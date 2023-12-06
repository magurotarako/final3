import sys
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


#いくつyamlファイルを作成するかをコマンドライン引数で指定（python3 make_yaml.py number alpha match borderの形で指定）
number = int(sys.argv[1])
alpha = float(sys.argv[2])
match = int(sys.argv[3])
border = int(sys.argv[4])

for i in range(number):
    seed_yaml = {'seed':i + 1, 'alpha':alpha, 'match':match, 'border':border}
    # data output to yaml
    output = dump(seed_yaml, Dumper=Dumper)
    with open("seed{:04d}_alpha{}_match{}_border{}.yml".format(i + 1, alpha, match, border), 'w') as yml:
        yml.write(output)

N = number

with open("task.sh", "w") as f:
    for i in range(N):
        script = "makeLogsFromSeed.py seed{:04d}_alpha{}_match{}_border{}.yml".format(i + 1, alpha, match, border)
        #result = "result{:03d}.dat".format(i)
        f.write("python3 {}\n".format(script))

print("You made some files.")
print("task.sh and seed{1~number}_alpha{1}_match{2}_border{3}.yml was made in this directory.")