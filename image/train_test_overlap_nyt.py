import matplotlib.pyplot as plt

a = """"
100%|█████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.01it/s]
precision: 0.1959, recall: 0.1215, fscore: 0.1500 ||
f1 =  0.14995640802107235
100%|█████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.38s/it]
precision: 0.2365, recall: 0.1542, fscore: 0.1866 ||
f1 =  0.1866493843163724
100%|█████████████████████████████████████████████████████████████████| 3/3 [00:02<00:00,  1.07it/s]
precision: 0.2661, recall: 0.1709, fscore: 0.2082 ||
f1 =  0.2081545064378532
100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.05s/it]
precision: 0.3067, recall: 0.1965, fscore: 0.2395 ||
f1 =  0.23953823953831269
100%|█████████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.13s/it]
precision: 0.3307, recall: 0.2153, fscore: 0.2608 ||
f1 =  0.26083112290015376
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.10it/s]
precision: 0.3531, recall: 0.2344, fscore: 0.2818 ||
f1 =  0.2817955112220048
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.07s/it]
precision: 0.3827, recall: 0.2557, fscore: 0.3066 ||
f1 =  0.3065522620905377
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.06it/s]
precision: 0.3998, recall: 0.2673, fscore: 0.3204 ||
f1 =  0.3203920090464138
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.00s/it]
precision: 0.4136, recall: 0.2773, fscore: 0.3320 ||
f1 =  0.33201581027672783
100%|█████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.05s/it]
precision: 0.4288, recall: 0.2921, fscore: 0.3475 ||
f1 =  0.3475274725275173
"""
# F1 scores
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
wdec = [0.1764, 0.2775, 0.3469, 0.4126, 0.4431, 0.472, 0.5047, 0.5249, 0.5359, 0.5497]
seq2umtree = [
    0.15,
    0.1866,
    0.2082,
    0.2395,
    0.2608,
    0.2818,
    0.3066,
    0.3204,
    0.332,
    0.3475,
]
selection = [t + 0.15 for t in wdec]

# fig = plt.figure()
# ax = plt.subplot(111)
# ax.plot(x, wdec, label='WDec')
# ax.plot(x, selection, label='MHS')
# ax.plot(x, seq2umtree, label='Seq2UMTree')
# fig.xlabel('Frequency threshold')
# fig.ylabel('F1')
# plt.title('NYT')
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),&nbsp; shadow=True, ncol=2)
# plt.show()


plt.plot(x, wdec, color="g", label="WDec")
plt.plot(x, seq2umtree, color="orange", label="Seq2UMTree")
# plt.plot(x, selection, color='blue',  label='Selection')

plt.xlabel("Frequency threshold")
plt.ylabel("F1")
plt.ylim(0, 1)
plt.legend()
plt.title("NYT")
plt.show()

# print(a.splitlines())
# result = []
# for line in a.splitlines():
#     if 'fscore' in line:
#         # print(line)
#         score = float(line[-10:-2])
#         result.append(score)
#         # print(score)
# print(result)
