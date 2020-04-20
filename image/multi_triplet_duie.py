import matplotlib.pyplot as plt

a = """
100%|███████████████████████████████████████████████████████████████| 25/25 [00:39<00:00,  1.41s/it]
precision: 0.5778, recall: 0.5677, fscore: 0.5727 ||
f1 =  0.572731932342393
100%|███████████████████████████████████████████████████████████████| 18/18 [00:40<00:00,  2.00s/it]
precision: 0.7631, recall: 0.7236, fscore: 0.7428 ||
f1 =  0.7428111509475398
100%|█████████████████████████████████████████████████████████████████| 6/6 [00:18<00:00,  2.98s/it]
precision: 0.7498, recall: 0.6658, fscore: 0.7053 ||
f1 =  0.7053002473948616
100%|█████████████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.10s/it]
precision: 0.7992, recall: 0.7039, fscore: 0.7485 ||
f1 =  0.7484960666358226
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:18<00:00,  4.28s/it]
precision: 0.8380, recall: 0.7386, fscore: 0.7852 ||
f1 =  0.7851783598789024
"""


wdec = [0.5607, 0.6806, 0.505, 0.4371, 0.3288]
seq2umtree = [0.5727, 0.7428, 0.7053, 0.7485, 0.7852]

# print(a.splitlines())
# result = []
# for line in a.splitlines():
#     if 'fscore' in line:
#         # print(line)
#         score = float(line[-10:-2])
#         result.append(score)
#         # print(score)
# print(result)
x = [1, 2, 3, 4, 5]
plt.plot(x, wdec, color="g", label="WDec")
plt.plot(x, seq2umtree, color="orange", label="Seq2UMTree")
# plt.plot(x, selection, color='blue',  label='Selection')

plt.xlabel("Triplet number per sentence")
plt.ylabel("F1")
plt.ylim(0, 1)
plt.xticks(range(1, 6), ["1", "2", "3", "4", ">4x"])

plt.legend()
# plt.title('DuIE')
plt.show()
