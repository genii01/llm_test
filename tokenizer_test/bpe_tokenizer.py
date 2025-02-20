from collections import defaultdict


def get_stats(tokens):
    """토큰 쌍 빈도를 계산하는 함수"""
    stats = defaultdict(int)
    for word in tokens:
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += 1
    return stats


def merge_tokens(tokens, merge_pair):
    """가장 많이 등장하는 문자 쌍을 병합하는 함수"""
    new_tokens = []
    for word in tokens:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == merge_pair:
                new_word.append(word[i] + word[i + 1])  # 병합
                i += 2  # 두 개의 문자를 병합했으므로 다음 문자로 이동
            else:
                new_word.append(word[i])
                i += 1
        new_tokens.append(new_word)
    return new_tokens


# 예제 단어들
tokens = [
    ["l", "o", "w", "e", "r"],
    ["l", "o", "w", "e", "s", "t"],
    ["n", "e", "w", "e", "s", "t"],
]
num_merges = 5  # 병합 횟수 제한

print("Initial Tokens:", tokens)

for _ in range(num_merges):
    stats = get_stats(tokens)
    if not stats:
        break
    best_pair = max(stats, key=stats.get)  # 가장 많이 등장하는 문자 쌍 찾기
    tokens = merge_tokens(tokens, best_pair)  # 병합
    print(f"Merge {best_pair}: {tokens}")
