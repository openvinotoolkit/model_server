import subprocess
 
subprocess.run(["/opt/kaldi/src/featbin/feat-to-len", "scp:/opt/data/feats.scp", "ark,t:feats_length.txt"])
 
f = open("ivector_online.1.ark.txt", "r")
g = open("ivector_online_ie.ark.txt", "w")
length_file = open("feats_length.txt", "r")
for line in f:
    if "[" not in line:
        for i in range(frame_count):
            line = line.replace("]", " ")
            g.write(line)
    else:
        g.write(line)
        frame_count = int(length_file.read().split(" ")[1])
g.write("]")
f.close()
g.close()
length_file.close()