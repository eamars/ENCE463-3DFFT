from subprocess import Popen, PIPE

fp = open("result.csv", "wb")

for i in range(1, 513):
	p = Popen(["3DFFT.exe", str(i)], stdout=PIPE)
	out = p.stdout.read()

	print(i)
	fp.write(out)
	fp.flush()

fp.close()