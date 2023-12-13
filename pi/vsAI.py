import sys
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


#いくつyamlファイルを作成するかをコマンドライン引数で指定（python3 vsAI.py number matchの形で指定）
number = int(sys.argv[1])
match = int(sys.argv[2])

with open("vsAI.sh", "w") as f:
    for i in range(number):
        f.write("python3 play_geister.py {} {}\n".format(i + 1, match))

