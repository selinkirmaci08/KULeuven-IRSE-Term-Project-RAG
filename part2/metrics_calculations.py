import numpy as np

def compute_metrics(query_list, all_ranked, idx_to_acl_id, top_k=5):
    precisions, recalls, f1s, average_precisions = [], [], [], []
    total_tp, total_retrieved, total_relevant = 0, 0, 0

    for i, query in enumerate(query_list):
        gold_ids = set(query["r"])
        if not gold_ids:
            continue

        ranked_ids = [idx_to_acl_id[idx] for idx in all_ranked[i][:top_k]]
        retrieved  = set(ranked_ids)

        tp = len(retrieved & gold_ids)
        precision = tp / len(retrieved) if retrieved else 0
        recall = tp / len(gold_ids)  if gold_ids  else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        hits, ap = 0, 0.0
        for rank, doc_id in enumerate(ranked_ids, start=1):
            if doc_id in gold_ids:
                hits += 1
                ap   += hits / rank
        average_precisions.append(ap / min(len(gold_ids), top_k))

        total_tp        += tp
        total_retrieved += len(retrieved)
        total_relevant  += len(gold_ids)

    macro_p  = np.mean(precisions)
    macro_r  = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    micro_p  = total_tp / total_retrieved if total_retrieved else 0
    micro_r  = total_tp / total_relevant  if total_relevant  else 0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0
    MAP      = np.mean(average_precisions)

    print(f"── Macro ── P: {macro_p:.4f}  R: {macro_r:.4f}  F1: {macro_f1:.4f}")
    print(f"── Micro ── P: {micro_p:.4f}  R: {micro_r:.4f}  F1: {micro_f1:.4f}")
    print(f"── MAP   ── {MAP:.4f}")
    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, MAP


def compute_metrics_from_acl_ids(query_list, ranked_acl_ids, top_k=10):
    precisions, recalls, f1s, average_precisions = [], [], [], []
    total_tp, total_retrieved, total_relevant = 0, 0, 0

    for i, query in enumerate(query_list):
        gold_ids  = set(query["r"])
        retrieved = set(ranked_acl_ids[i][:top_k])

        tp = len(retrieved & gold_ids)
        precision = tp / len(retrieved) if retrieved else 0
        recall = tp / len(gold_ids)  if gold_ids  else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        hits, ap = 0, 0.0
        for rank, doc_id in enumerate(ranked_acl_ids[i][:top_k], start=1):
            if doc_id in gold_ids:
                hits += 1
                ap   += hits / rank
        average_precisions.append(ap / min(len(gold_ids), top_k))

        total_tp += tp
        total_retrieved += len(retrieved)
        total_relevant  += len(gold_ids)

    macro_p  = np.mean(precisions)
    macro_r  = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    micro_p  = total_tp / total_retrieved if total_retrieved else 0
    micro_r  = total_tp / total_relevant  if total_relevant  else 0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0
    MAP      = np.mean(average_precisions)

    print(f"── Macro ── P: {macro_p:.4f}  R: {macro_r:.4f}  F1: {macro_f1:.4f}")
    print(f"── Micro ── P: {micro_p:.4f}  R: {micro_r:.4f}  F1: {micro_f1:.4f}")
    print(f"── MAP   ── {MAP:.4f}")
    return macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, MAP