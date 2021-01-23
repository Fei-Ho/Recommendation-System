import math

# precsion@N
def cal_PrecN(ranked_list, ground_truth, N):
    """
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    """
    ranked_list = ranked_list[:N]
    hits = 0
    for i in range(len(ranked_list)):
        if ranked_list[i] in ground_truth:
            hits += 1
    return hits/N

# recall@N
def cal_RecallN(ranked_list, ground_truth, N):
    """
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    """
    ranked_list = ranked_list[:N]
    hits = 0
    for i in range(len(ranked_list)):
        if ranked_list[i] in ground_truth:
            hits += 1
    return hits/len(ground_truth)

# F-score
def cal_FScore(prec, recall, alpha=1):
    return (1+alpha*alpha)*prec*recall/(alpha*alpha*prec+recall)

# 单个用户的HR@N
def cal_puHitRateN(ranked_list, ground_truth, N):
    '''
    Hit Rate: 只要ranked_list中有任意一个出现在ground_truth，即为命中，返回1，否则为0
    当ground_truth的个数只有一个时，hit rate和recall指标一致
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    '''
    ranked_list = ranked_list[:N]
    hits = 0
    if len(set(ground_truth).intersection(ranked_list))!=0:
        return 1
    else:
        return 0

# AP
def cal_AP(ranked_list, ground_truth):
    """
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    AP: 平均准确率
    mAP: 计算每个用户的AP, 再求平均
    """
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0) # prec@N的值：hits/(n+1.0)，将每个位置上的prec@N累加
    if hits > 0:
        return sum_precs / len(ground_truth) # 返回平均准确率
    else:
        return 0

# 单个用户的MRR
def cal_puMRR(ranked_list, ground_truth):
    """
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    """
    for i in range(len(ranked_list)):
        # ranked_list中第一个在ground-truth中出现的item所在的位置
        if ranked_list[i] in ground_truth:
            return 1/i
    return 0

# DCG
def cal_DCGN(ranked_list, ground_truth, N):
    """
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    """
    ranked_list = ranked_list[:N]
    DCG = 0
    for i in range(len(ranked_list)):
        if ranked_list[i] in ground_truth:
            rel_i = 1 # 每个位置的效益rel都为1,或者为2^rel-1
            DCG += rel_i/math.log(2,1 + (i+1))  
    return DCG

# NDCG
def cal_NDCGN(ranked_list, ground_truth, N):
    """
    ranked_list: 根据概率排序的item id列表
    ground_truth: 真实交互item id列表
    """
    ranked_list = ranked_list[:N]
    DCG = 0
    IDCG = 0
    num_rel = 0
    for i in range(len(ranked_list)):
        if ranked_list[i] in ground_truth:
            num_rel += 1
            rel_i = 1 # 每个位置的效益rel都为1,或者为2^rel-1
            DCG += rel_i/math.log(2, 1 + (i+1))
    for i in range(num_rel):
        rel_i = 1
        IDCG += rel_i/math.log(2,1+(i+1))
    return DCG/IDCG