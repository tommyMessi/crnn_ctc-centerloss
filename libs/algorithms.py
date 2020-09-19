import numpy as np


def edit_distance(src, dst, normalize=True):
    """
    http://www.dreamxu.com/books/dsa/dp/edit-distance.html
    https://en.wikipedia.org/wiki/Levenshtein_distance
    https://www.quora.com/How-do-I-figure-out-how-to-iterate-over-the-parameters-and-write-bottom-up-solutions-to-dynamic-programming-related-problems/answer/Michal-Danil%C3%A1k?srid=3OBi&share=1

    编辑距离(Levenshtein distance 莱文斯坦距离)
    给定 2 个字符串 a, b. 编辑距离是将 a 转换为 b 的最少操作次数，操作只允许如下 3 种：

    1. 插入一个字符，例如：fj -> fxj
    2. 删除一个字符，例如：fxj -> fj
    3. 替换一个字符，例如：jxj -> fyj
    """

    m = len(src)
    n = len(dst)

    # 初始化二位数组，保存中间值。多一维可以用来处理 src/dst 为空字符串的情况
    # d[i, j] 表示 src[0,i] 与 dst[0,j] 之间的距离
    d = np.zeros((n + 1, m + 1))
    #     print(d.shape)

    # 第一列赋值
    for i in range(1, n + 1):
        d[i][0] = i

    # 第一行赋值
    for j in range(1, m + 1):
        d[0][j] = j

    for j in range(1, m + 1):
        for i in range(1, n + 1):
            if src[j - 1] == dst[i - 1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(d[i - 1, j] + 1,
                          d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)

    distance = d[-1][-1]

    if normalize:
        if len(src) == 0 and len(dst) == 0:
            return 0

        return distance / len(src)
    else:
        return distance


if __name__ == "__main__":
    assert edit_distance("kitten", "sitting", False) == 3
    assert edit_distance('ebab', 'abcd', False) == 3
    assert edit_distance('1234', '', False) == 4
    assert edit_distance('kilo', 'kilogram', False) == 4

    assert edit_distance('kilo', '') == 1
    assert edit_distance('kilo', 'kil') == 1/4
    assert edit_distance('kilo11', 'kil') == 3/6
