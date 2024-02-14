# 决策树的节点
class DecisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        # 待检验的判断条件对应的列索引值
        self.col = col
        # 使结果为True的当前列必须匹配的值
        self.value = value
        # 字典，保存当前分支的结果
        self.results = results
        # 结果为True时，树上相对于当前节点的子树上的节点
        self.tb = tb
        # 结果为False时，树上相对于当前节点的子树上的节点
        self.fb = fb


# 对y的各种可能的取值出现的个数进行计数
def unique_counts(rows):
    results = {}
    for row in rows:
        r = row[len(row) - 1]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


# 熵
def entropy(rows):
    from math import log
    results = unique_counts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent -= p * log(p, 2)
    return ent


def divide_set(rows, column, value):
    """
    根据指定列的值将数据行分割成两个集合
    :param rows: 数据行列表
    :param column: 指定用于分割的列的索引
    :param value: 分割依据的值
    :return: set1, set2，分割后的两个数据集
    """
    if isinstance(value, int) or isinstance(value, float):
        split_function = lambda row: row[column] >= value
    else:
        split_function = lambda row: row[column] == value
    # 创建两个新的数据集
    set1 = [row for row in rows if split_function(row)]
    set2 = [row for row in rows if not split_function(row)]
    return set1, set2


# 计算基尼不纯度
def gini_impurity(rows):
    total = len(rows)
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    imp = 0
    for label in counts:
        prob = counts[label] / total
        imp += prob * (1 - prob)
    return imp


# 递归地创建树
def build_tree(rows, scoref=entropy):
    if len(rows) == 0:
        return DecisionNode()
    current_score = scoref(rows)
    # 定义一些变量以记录最佳拆分条件
    best_gain = 0.0
    best_criteria = None
    best_sets = None
    column_count = len(rows[0]) - 1
    for col in range(0, column_count):
        # 在当前列中生成一个由不同值构成的序列
        column_values = {}
        for row in rows:
            column_values[row[col]] = 1  # 初始化
        # 根据这一列中的每个值，尝试对数据集进行拆分
        for value in column_values.keys():
            (set1, set2) = divide_set(rows, col, value)

            # 信息增益
            p = float(len(set1)) / len(rows)
            gain = current_score - p * scoref(set1) - (1 - p) * scoref(set2)
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                best_gain = gain
                best_criteria = (col, value)
                best_sets = (set1, set2)

    # 创建子分支
    if best_gain > 0:
        trueBranch = build_tree(best_sets[0])
        falseBranch = build_tree(best_sets[1])
        # 递归调用
        return DecisionNode(col=best_criteria[0], value=best_criteria[1],
                            tb=trueBranch, fb=falseBranch)
    else:
        # 递归调用
        return DecisionNode(results=unique_counts(rows))


# 决策树的显示
def print_tree(tree, indent='   '):
    # 是否是叶节点
    if tree.results is not None:
        print(str(tree.results))
    else:
        # 打印判断条件
        print(str(tree.col) + ":" + str(tree.value) + "? ")
        # 打印分支
        print(indent + "T->"),
        print_tree(tree.tb, indent + " ")
        print(indent + "F->"),
        print_tree(tree.fb, indent + " ")


def classify(observation, tree):
    if tree.results is not None:
        return tree.results
    else:
        v = observation[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


if __name__ == '__main__':
    my_data = [['high', 'heavy', 'flack', 'normal', 'bubble-like', 'Pneumonia'],
               ['medium', 'heavy', 'flack', 'normal', 'bubble-like', 'Pneumonia'],
               ['low', 'slight', 'spot', 'normal', 'dry-beep', 'Pneumonia'],
               ['high', 'medium', 'flack', 'normal', 'bubble-like', 'Pneumonia'],
               ['medium', 'slight', 'flack', 'normal', 'bubble-like', 'Pneumonia'],
               ['absent', 'slight', 'strip', 'normal', 'normal', 'Tuberculosis'],
               ['high', 'heavy', 'hole', 'fast', 'dry-beep', 'Tuberculosis'],
               ['low', 'slight', 'strip', 'normal', 'normal', 'Tuberculosis'],
               ['absent', 'slight', 'spot', 'fast', 'dry-beep', 'Tuberculosis'],
               ['low', 'medium', 'flack', 'fast', 'normal', 'Tuberculosis']
               ]
    # 分割数据集
    divide_set(my_data, 5, 'Pneumonia')
    # 计算基尼指数
    gini_impurity(my_data)
    # 创建树
    my_tree = build_tree(my_data)
    # 打印树
    print_tree(my_tree)
    # 测试数据
    # classify(['high', 'heavy', 'flack', 'normal', 'bubble-like'], my_tree)
