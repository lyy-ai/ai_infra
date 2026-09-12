# MinHash + LSH 模糊去重教学演示（16.2 节配套）
# 运行: python minhash_dedup_demo.py
# 只依赖标准库 + numpy
#
# 演示流程：文档 -> n-gram 集合 -> MinHash 签名 -> LSH 分桶候选 -> Jaccard 验证
# 可以直观看到：MinHash 签名相似度 ≈ Jaccard 相似度，LSH 把 O(N^2) 比较降为近线性

import numpy as np

rng = np.random.default_rng(7)

N_HASH = 100      # MinHash 签名维数
N_BANDS = 20      # LSH band 数（每 band 5 维）
PRIME = (1 << 31) - 1


def shingles(text, n=3):
    tokens = text.split()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def minhash_sig(shingle_set, hash_funcs):
    sig = np.full(N_HASH, np.iinfo(np.int64).max)
    for s in shingle_set:
        h = hash(s)
        for i, (a, b) in enumerate(hash_funcs):
            v = (a * h + b) % PRIME
            if v < sig[i]:
                sig[i] = v
    return sig


def jaccard(a, b):
    return len(a & b) / len(a | b)


def main():
    base = "the quick brown fox jumps over the lazy dog and runs through the forest"
    docs = [
        base,                                                     # 0 原文
        "the quick brown fox jumps over the lazy dog and runs",   # 1 近重复（删掉尾部）
        base + " again",                                          # 2 近重复（加尾部）
        "a completely different document about machine learning", # 3 无关文档
        "neural networks and deep learning are transforming ai",  # 4 无关文档
        "the quick brown fox jumps over a lazy dog and runs through the forest",  # 5 微调近重复
    ]
    shingle_sets = [shingles(d) for d in docs]

    hash_funcs = [(int(rng.integers(1, PRIME)), int(rng.integers(0, PRIME))) for _ in range(N_HASH)]
    sigs = [minhash_sig(s, hash_funcs) for s in shingle_sets]

    print("=" * 80)
    print("MinHash 签名相似度 vs 真实 Jaccard（验证近似性）")
    print("=" * 80)
    print(f"{'文档对':<14}{'签名相似度':>12}{'Jaccard':>10}")
    pairs = [(0, 1), (0, 2), (0, 3), (0, 5), (3, 4)]
    for i, j in pairs:
        sig_sim = np.mean(sigs[i] == sigs[j])
        print(f"doc{i} vs doc{j:<5}{sig_sim:>12.2f}{jaccard(shingle_sets[i], shingle_sets[j]):>10.2f}")

    print()
    print("=" * 80)
    print(f"LSH 分桶（{N_BANDS} bands × {N_HASH // N_BANDS} rows），找近重复候选对")
    print("=" * 80)
    rows = N_HASH // N_BANDS
    buckets = {}
    for doc_id, sig in enumerate(sigs):
        for b in range(N_BANDS):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets.setdefault(key, []).append(doc_id)
    candidates = set()
    for ids in buckets.values():
        for x in ids:
            for y in ids:
                if x < y:
                    candidates.add((x, y))
    print(f"候选对（同 bucket 碰撞）: {sorted(candidates)}")
    print(f"暴力对比需要 {len(docs) * (len(docs) - 1) // 2} 对，LSH 只需验证 {len(candidates)} 对")
    threshold = 0.5
    for i, j in sorted(candidates):
        sim = jaccard(shingle_sets[i], shingle_sets[j])
        verdict = "重复 -> 删除其一" if sim >= threshold else "保留"
        print(f"  doc{i} vs doc{j}: Jaccard={sim:.2f} -> {verdict}")

    print()
    print("读法：")
    print("1. MinHash 签名相似度逼近 Jaccard，维数越高越准（代价是签名存储）")
    print("2. LSH 把全量两两比较降为只验证碰撞候选——十亿文档也可行")
    print("3. band 数控制松紧：band 多 -> 召回高候选多；真实管线按阈值(如0.8)调参")


if __name__ == "__main__":
    main()
