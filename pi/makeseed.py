N = 19

for i in range(N):
    filename = "seed{:02d}.dat".format(i)
    print(filename)
    with open(filename, "w") as f:
        f.write(str(i))

with open("task.sh", "w") as f:
    print("task.sh")
    for i in range(N):
        filename = "seed{:02d}.dat".format(i)
        result = "result{:02d}.dat".format(i)
        f.write("python pi.py < {} > {}\n".format(filename, result))
