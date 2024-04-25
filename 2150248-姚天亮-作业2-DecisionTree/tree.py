import math

def watermelon3():
    """
    创建测试的数据集，里面的数值中具有连续值
    :return:
    """
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.460, '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.376, '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.264, '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.318, '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.215, '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.237, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.149, '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.211, '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.091, '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.267, '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.057, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.099, '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.161, '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.198, '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.370, '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.042, '坏瓜']
    ]
 
    # 特征值列表
    labels = ['色泽', '根蒂', '敲击', '纹理', '脐部', '触感', '含糖率']
 
    # 特征对应的所有可能的情况
    labels_full = {}
 
    for i in range(len(labels)):
        labelList = [example[i] for example in dataSet]
        uniqueLabel = set(labelList)
        labels_full[labels[i]] = uniqueLabel
 
    return dataSet, labels, labels_full

class TreeNode:
    """
    决策树结点类
    """
    current_index = 0
 
    def __init__(self, parent=None, attr_name=None, children=None, judge=None, split=None, data_index=None,
                 attr_value=None, rest_attribute=None):
        self.parent = parent  # 父节点，根节点的父节点为 None
        self.attribute_name = attr_name  # 本节点上进行划分的属性名
        self.attribute_value = attr_value  # 本节点上划分属性的值，是与父节点的划分属性名相对应的
        self.children = children  # 孩子结点列表
        self.judge = judge  # 如果是叶子结点，需要给出判断
        self.split = split  # 如果是使用连续属性进行划分，需要给出分割点
        self.data_index = data_index  # 对应训练数据集的训练索引号
        self.index = TreeNode.current_index  # 当前结点的索引号，方便输出时查看
        self.rest_attribute = rest_attribute  # 尚未使用的属性列表
        TreeNode.current_index += 1
 
    def to_string(self):
        this_string = '当前节点 : ' + str(self.index) + ";\n"
        if not (self.parent is None):
            parent_node = self.parent
            this_string = this_string + '父节点 : ' + str(parent_node.index) + ";\n"
            this_string = this_string + str(parent_node.attribute_name) + " : " + str(self.attribute_value) + ";\n"
        this_string = this_string + "满足数据 : " + str(self.data_index) + ";\n"
        if not(self.children is None):
            this_string = this_string + '判断依据 : ' + str(self.attribute_name) + ";\n"
            child_list = []
            for child in self.children:
                child_list.append(child.index)
            this_string = this_string + '子节点 : ' + str(child_list)
        if not (self.judge is None):
            this_string = this_string + '叶子标签 : ' + self.judge
        return this_string
 
 
def is_number(s):
    """判断一个字符串是否为数字"""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        pass
    return False
 
def ent(labels):
    """
    样本集合的信息熵
    :param labels: 样本集合中数据的类别标签
    :return:
    """
    label_count = {}
 
    for item in labels:
        if item not in label_count:
            label_count[item] = 1
        else:
            label_count[item] += 1
 
    n = len(labels)
    entropy = 0.0
    for item in label_count.values():
        p = item / n
        entropy -= p*math.log(p, 2)
 
    return entropy
 
def gain(attribute, labels, is_value=False):
    """
    计算信息增益
    :param attribute: 集合中样本该属性的值列表
    :param labels: 集合中样本的数据标签
    :return:
    """
    info_gain = ent(labels)
    n = len(labels)
    split_value = None  # 如果是连续值的话，也需要返回分隔界限的值
 
    if is_value:
        sorted_attribute = attribute.copy()
        sorted_attribute.sort()
        split = []  # 候选的分隔点
        for i in range(0, n-1):
            temp = (sorted_attribute[i] + sorted_attribute[i+1]) / 2
            split.append(temp)
        info_gain_list = []
        for temp_split in split:
            low_labels = []
            high_labels = []
            for i in range(0, n):
                if attribute[i] <= temp_split:
                    low_labels.append(labels[i])
                else:
                    high_labels.append(labels[i])
            temp_gain = info_gain - len(low_labels)/n*ent(low_labels) - len(high_labels)/n*ent(high_labels)
            info_gain_list.append(temp_gain)

        info_gain = max(info_gain_list)
        max_index = info_gain_list.index(info_gain)
        split_value = split[max_index]
    else:
        attribute_dict = {}
        label_dict = {}
        index = 0
        for item in attribute:
            if attribute_dict.__contains__(item):
                attribute_dict[item] = attribute_dict[item] + 1
                label_dict[item].append(labels[index])
            else:
                attribute_dict[item] = 1
                label_dict[item] = [labels[index]]
            index += 1

        for key, value in attribute_dict.items():
            info_gain = info_gain - value/n * ent(label_dict[key])

    return info_gain, split_value

def gain_ratio(attribute, labels, is_value=False):
    """
    计算信息增益比
    :param attribute: 集合中样本该属性的值列表
    :param labels: 集合中样本的数据标签
    :return:
    """
    gain_val, split_value = gain(attribute, labels, is_value)
    n = len(labels)
    intrinsic_val = 0.0
    if is_value:
        sorted_attribute = attribute.copy()
        sorted_attribute.sort()
        split = []
        for i in range(0, n - 1):
            temp = (sorted_attribute[i] + sorted_attribute[i + 1]) / 2
            split.append(temp)
        intrinsic_list = []
        for temp_split in split:
            low_count = 0
            high_count = 0
            for i in range(0, n):
                if attribute[i] <= temp_split:
                    low_count += 1
                else:
                    high_count += 1
            p_low = low_count / n
            p_high = high_count / n
            temp_intrinsic = 0.0
            if p_low != 0:
                temp_intrinsic -= p_low * math.log(p_low, 2)
            if p_high != 0:
                temp_intrinsic -= p_high * math.log(p_high, 2)
            intrinsic_list.append(temp_intrinsic)
        intrinsic_val = max(intrinsic_list)
    else:
        attribute_count = {}
        for item in attribute:
            if item not in attribute_count:
                attribute_count[item] = 1
            else:
                attribute_count[item] += 1
        for key, value in attribute_count.items():
            p = value / n
            intrinsic_val -= p * math.log(p, 2)

    if intrinsic_val == 0:
        return 0
    else:
        return gain_val / intrinsic_val
