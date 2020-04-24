import matplotlib.pyplot as plt

a = """
precision: 0.6125, recall: 0.5496, fscore: 0.5793 ||
f1 =  0.5793473401877891
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.44s/it]
precision: 0.6198, recall: 0.3251, fscore: 0.4265 ||
f1 =  0.426523297491245
100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.04it/s]
precision: 0.7213, recall: 0.2353, fscore: 0.3548 ||
f1 =  0.3548387096779397
"""


wdec = [0.4930, 0.3622, 0.3504]
seq2umtree = [0.3708, 0.2679, 0.3203]
selection = [0.5793, 0.4267, 0.3548]
# print(a.splitlines())
# result = []
# for line in a.splitlines():
#     if 'fscore' in line:
#         # print(line)
#         score = float(line[-10:-2])
#         result.append(score)
#         # print(score)
# print(result)
x = [1, 2, 3]
plt.plot(x, wdec, color="g", label="WDec")
plt.plot(x, seq2umtree, color="orange", label="Seq2UMTree")
plt.plot(x, selection, color='blue',  label='MHS')

plt.xlabel("Triplet number per sentence")
plt.ylabel("F1")
plt.ylim(0, 1)
# plt.xlim(0, 1)
plt.xticks(range(1, 4), ["1", "2", ">2"])

plt.legend()
# plt.title('DuIE')
plt.show()
