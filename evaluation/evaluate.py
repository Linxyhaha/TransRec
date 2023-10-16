import numpy as np 
import torch
import math
import time
import json
import tqdm
import ipdb

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
    
def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        user_length = 0
        cnt = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(predictedIndices[i])<topN[index]:
                cnt += 1
                continue
            if len(GroundTruth[i]) != 0:
                user_length += 1
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / user_length, 4))
        recall.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNdcg / user_length, 4))
        MRR.append(round(sumForMRR / user_length, 4))
        
    return precision, recall, NDCG, MRR

def group_computeTopNAccuracy(GroundTruth, predictedIndices, groupIndices, topN):
    """
        groupIndices is a list that contain the group idx in accordance with the GroundTruth list and predictedIndices list.
    """
    num_group = max(groupIndices)+1
    print(f"num of predicted samples: {len(predictedIndices)}")

    precision = [[] for _ in range(num_group)] 
    recall = [[] for _ in range(num_group)] 
    NDCG = [[] for _ in range(num_group)] 
    MRR = [[] for _ in range(num_group)] 

    for index in range(len(topN)):
        sumForPrecision = [0] * num_group
        sumForRecall = [0] * num_group
        sumForNdcg = [0] * num_group
        sumForMRR = [0] * num_group
        user_length = [0] * num_group

        for i in range(len(predictedIndices)):  
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)

                sumForPrecision[groupIndices[i]] += userHit / topN[index]
                sumForRecall[groupIndices[i]] += userHit / len(GroundTruth[i])               
                sumForNdcg[groupIndices[i]] += ndcg
                sumForMRR[groupIndices[i]] += userMRR
                user_length[groupIndices[i]] += 1

        for g_idx in range(num_group):
            try:
                precision[g_idx].append(round(sumForPrecision[g_idx] / user_length[g_idx], 4))
                recall[g_idx].append(round(sumForRecall[g_idx] / user_length[g_idx], 4))
                NDCG[g_idx].append(round(sumForNdcg[g_idx] / user_length[g_idx], 4))
                MRR[g_idx].append(round(sumForMRR[g_idx] / user_length[g_idx], 4))
            except:
                print(f"top {topN[index]}: group {g_idx} has no valid users.")

    return [(precision[g_idx], recall[g_idx], NDCG[g_idx], MRR[g_idx]) for g_idx in range(num_group)]
    

def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='beauty', type=str)
    args = parser.parse_args()
    print(args)

    with open(f"./output/{args.dataset}_output.json", "r") as f:
        gen = json.load(f)
        all_pred_list = []
        for q_idx, retrieves in enumerate(tqdm.tqdm(gen)):
            pred_list = []
            for each_retrieve in retrieves['ctxs']:
                pred_list.append(each_retrieve['item_id'])
            all_pred_list.append(pred_list)

    all_gold_list = []
    with open(f'./data/{args.dataset}/rec_data/sequential_data.txt','r') as f:
        for line in f:
            line = line.strip() 
            all_gold_list.append([line.split(' ')[-1]])
        
    test_results = computeTopNAccuracy(all_gold_list, all_pred_list, [5,10])
    print("Overall Results")
    print_results(None, None, test_results)            