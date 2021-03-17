def merge(fins, fout):
    samples = []

    for fin in fins:
        with open(fin) as f:
            samples_part = f.read().strip().split("\n\n")
            samples += samples_part
            print(fin, len(samples_part))
    
    with open(fout, "w") as w:
        print(fout, len(samples))
        for sample in samples:
            w.write(sample + "\n")
            w.write("\n")

merge(["train1.txt", "train2.txt"], "train.txt")
merge(["valid1.txt", "valid2.txt"], "valid.txt")
merge(["test1.txt", "test2.txt"], "test.txt")

