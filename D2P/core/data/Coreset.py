import torch
import numpy as np
from .sampling import GraphDensitySampler
from tqdm import tqdm


def bin_allocate(num, bins, mode='uniform', initial_budget=None):
    sorted_index = torch.argsort(bins)
    sort_bins = bins[sorted_index]

    num_bin = bins.shape[0]

    rest_exp_num = num
    budgets = []
    for i in range(num_bin):
        if sort_bins[i] == 0:
            budgets.append(0)
            continue
        # rest_bins = num_bin - i
        rest_bins = torch.count_nonzero(sort_bins[i:])
        if mode == 'uniform':
            avg = rest_exp_num // rest_bins
            cur_num = min(sort_bins[i].item(), avg)
            rest_exp_num -= cur_num
        else:
            avg = initial_budget[sorted_index[i]]
            cur_num = min(sort_bins[i].item(), avg)
            delta = int((avg - cur_num)/max(1, (rest_bins - 1)))
            # print("At index %s, changing budget from %s to %s and reallocating %s to %s bins" % (i, avg, cur_num, delta, rest_bins-1))
            for j in range(i+1, num_bin):
                initial_budget[sorted_index[j]] += delta
        budgets.append(cur_num)

    budgets = torch.tensor(budgets)
    if torch.sum(budgets) < num: # TODO: check again
        delta = num - torch.sum(budgets)
        i = 1
        while delta and i <= num_bin:
            if budgets[-i] < sort_bins[-i]:
                budgets[-i] += 1
                delta -= 1
            i += 1

    rst = torch.zeros((num_bin,)).type(torch.int)
    rst[sorted_index] = torch.tensor(budgets).type(torch.int)

    assert all([b<= r for r, b in zip(bins, rst)]), ([(r.item(),b.item()) for r, b in zip(bins, rst)], bins, [x.item() for x in torch.tensor(budgets)[sorted_index]])
    return rst


class CoresetSelection(object):
    @staticmethod
    def stratified_sampling(variance, coreset_num, data_embeds=None, n_neighbor=5, stratas=5):
        graph = GraphDensitySampler(X=data_embeds, importance_scores=variance,
                                        n_neighbor=n_neighbor, graph_mode='sum',
                                        graph_sampling_mode='weighted'
                                        # precomputed_dists=args.precomputed_dists,
                                        # precomputed_neighbors=args.precomputed_neighbors
                                    )
        score = torch.tensor(graph.graph_density)

        min_score = torch.min(score)
        max_score = torch.max(score) * 1.0001
        print("Min score: %s, max score: %s" % (min_score.item(), max_score.item()))
        step = (max_score - min_score) / stratas

        def bin_range(k):
            return min_score + k * step, min_score + (k + 1) * step

        strata_num = []
        ##### calculate number of samples in each strata #####
        for i in range(stratas):
            start, end = bin_range(i)
            num = torch.logical_and(score >= start, score < end).sum()
            strata_num.append(num)
        strata_num = torch.tensor(strata_num)


        budgets = bin_allocate(coreset_num, strata_num)
        # assert budgets.sum().item() == coreset_num, (budgets.sum(), coreset_num)
        print(budgets, budgets.sum())

        ##### sampling in each strata #####
        selected_index = []
        sample_index = torch.arange(variance.shape[0])

        pools, kcenters = [], []
        for i in tqdm(range(stratas), desc='sampling from each strata'):
            start, end = bin_range(i)
            mask = torch.logical_and(score >= start, score < end)
            pool = sample_index[mask]
            pools.append(pool)

            if pool.shape[0] <= n_neighbor: # if num of samples are less than size of graph, select all
                rand_index = torch.randperm(pool.shape[0])
                selected_idxs = rand_index[:budgets[i]].numpy().tolist()
            else:
                sampling_method = GraphDensitySampler(X=None if data_embeds is None else data_embeds[pool],
                                                          importance_scores=score[pool],
                                                          n_neighbor=n_neighbor, graph_mode='sum',
                                                          graph_sampling_mode='weighted'
                                                      # precomputed_dists=args.precomputed_dists,
                                                      # precomputed_neighbors=args.precomputed_neighbors
                                                      )
                selected_idxs = sampling_method.select_batch_(budgets[i])

            kcenters.append(pool[selected_idxs])

        for samples in kcenters:
            selected_index += samples

        selected_index = [tensor.item() for tensor in selected_index]
        return selected_index