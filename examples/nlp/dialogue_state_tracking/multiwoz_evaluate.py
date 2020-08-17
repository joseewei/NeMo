import math
from collections import Counter

from nltk.util import ngrams


## BLEU score calculator via GentScorer interface
## it calculates the BLEU-4 by taking the entire corpus in
## Calulate based multiple candidates against multiple references
def score(hypothesis, corpus, n=1):
    # containers
    count = [0, 0, 0, 0]
    clip_count = [0, 0, 0, 0]
    r = 0
    c = 0
    weights = [0.25, 0.25, 0.25, 0.25]

    # accumulate ngram statistics
    for hyps, refs in zip(hypothesis, corpus):
        # if type(hyps[0]) is list:
        #    hyps = [hyp.split() for hyp in hyps[0]]
        # else:
        #    hyps = [hyp.split() for hyp in hyps]

        # refs = [ref.split() for ref in refs]
        hyps = [hyps]
        # Shawn's evaluation
        # refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
        # hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

        for idx, hyp in enumerate(hyps):
            for i in range(4):
                # accumulate ngram counts
                hypcnts = Counter(ngrams(hyp, i + 1))
                cnt = sum(hypcnts.values())
                count[i] += cnt

                # compute clipped counts
                max_counts = {}
                for ref in refs:
                    refcnts = Counter(ngrams(ref, i + 1))
                    for ng in hypcnts:
                        max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                clipcnt = dict((ng, min(count, max_counts[ng])) for ng, count in hypcnts.items())
                clip_count[i] += sum(clipcnt.values())

            # accumulate r & c
            bestmatch = [1000, 1000]
            for ref in refs:
                if bestmatch[0] == 0:
                    break
                diff = abs(len(ref) - len(hyp))
                if diff < bestmatch[0]:
                    bestmatch[0] = diff
                    bestmatch[1] = len(ref)
            r += bestmatch[1]
            c += len(hyp)
            if n == 1:
                break
    # computing bleu score
    p0 = 1e-7
    bp = 1 if c > r else math.exp(1 - float(r) / float(c))
    p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bleu = bp * math.exp(s)
    return bleu


# def get_err(generated_text, ref_)

# def get_success_inform(generate, true):
#     # SUCCESS
#     real_requestables = []
#     provided_requestables = [] - from the NLG
#     if match == 1.0:
#         for domain in domains_in_goal:
#             success_stat = 0
#             domain_success = 0
#             if len(real_requestables[domain]) == 0:
#                 success += 1
#                 success_stat = 1
#                 stats[domain][1] = success_stat
#                 print ('---> in suc 1:')
#                 continue
#             # if values in sentences are super set of requestables
#             for request in set(provided_requestables[domain]):
#                 if request in real_requestables[domain]:
#                     domain_success += 1
#
#             if domain_success >= len(real_requestables[domain]):
#                 success += 1
#                 success_stat = 1
#
#             stats[domain][1] = success_stat
#
#         # final eval
#         if soft_acc:
#             success = float(success)/len(real_requestables)
#         else:
#             if success >= len(real_requestables):
#                 success = 1
#             else:
#                 success = 0
#
#     # rint requests, 'DIFF', requests_real, 'SUCC', success
#     return success, match, stats
