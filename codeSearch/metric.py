import numpy as np 
def top(real, predict):
    top1 = []
    top3 = []
    top5 = []
    top10 = []

    for i in range(len(real)):
        try:
            index = np.argwhere(predict[i]==real[i])[0][0]
            index += 1
        except Exception:
            top1.append(0)
            top3.append(0)
            top5.append(0)
            top10.append(0)
            continue

        if index > 10:
            top1.append(0)
            top3.append(0)
            top5.append(0)
            top10.append(0)
            continue

        if index <= 10:
            top10.append(1)
        else:
            top10.append(0)

        if index <= 5:
            top5.append(1)
        else:
            top5.append(0)

        if index <= 3:
            top3.append(1)
        else:
            top3.append(0)

        if index <= 1:
            top1.append(1)
        else:
            top1.append(0)

    return sum(top1)/len(top1), sum(top3)/len(top3), sum(top5)/len(top5), sum(top10)/len(top10)
