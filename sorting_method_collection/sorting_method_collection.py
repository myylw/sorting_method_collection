avg = [5, 2, 7, 6, 9, 0, 8, 1, 2, 4, 3, 7]
bad = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
good = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def mycounter():
    count = 0

    def _counter():
        nonlocal count
        count += 1
        return count

    return _counter


########################################  冒泡排序  ####################################
# 基础冒泡排序,冒泡排序属于交换排序
def bubble_sort(seq: list):
    for i in range(len(seq), 0, -1):
        for j in range(i - 1):
            if seq[j] > seq[j + 1]:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
    return seq


# 改良冒泡排序
# 当一轮未发生交换,则认为排序完毕
def bubble_sort_plus(seq: list):
    for i in range(len(seq), 0, -1):
        flag = False
        for j in range(i - 1):
            if seq[j] > seq[j + 1]:
                flag = True
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
        if flag is False:
            break
    return seq


# print(bubble_sort(avg))
# print(bubble_sort_plus(avg))

# 冒泡排序平均复杂度O(n^2),最佳情况O(n),最差情况O(n^2)
# 空间复杂度O(1)

########################################  选择排序  ####################################

def selection_sort(seq):
    for i in range(len(seq)):
        current, min_index = seq[i], i
        for j in range(i + 1, len(seq)):
            if seq[j] < current:
                min_index = j
                current = seq[j]
        seq[i], seq[min_index] = seq[min_index], seq[i]
    return seq


# print(selection_sort(avg))

# 选择排序平均复杂度O(n^2),最佳情况O(n^2),最差情况O(n^2)
# 空间复杂度O(1)

########################################  插入排序  ####################################

def insertion_sort(seq):
    for i in range(1, len(seq)):
        temp = seq[i]
        for j in range(i - 1, -2, -1):
            if temp < seq[j]:
                seq[j + 1] = seq[j]
            else:
                break
        seq[j + 1] = temp
    return seq


# print(insertion_sort(avg))


# 插入排序平均复杂度O(n^2),最佳情况O(n^2),最差情况O(n^2)
# 空间复杂度O(1)

########################################  侏儒排序  ####################################

def gnome_sort(seq):
    index = 0
    while index < len(seq) - 1:
        # print(seq)
        # print('   ' * index, '^')
        if index == 0 or seq[index] >= seq[index - 1]:
            index += 1
        if seq[index] < seq[index - 1]:
            seq[index], seq[index - 1] = seq[index - 1], seq[index]
            index -= 1

    return seq


# print(gnome_sort(avg))
# print(gnome_sort(good))

# 侏儒排序平均复杂度O(n^2),最佳情况O(n),最差情况O(n^2),稳定
# 空间复杂度O(1)

########################################  希尔排序  ####################################
# 希尔排序则为插入排序的改良版本,先将数据按照不同的步长分组排序,最后在进行插入排序
def shell_sort(seq):
    seq_step = len(seq) // 2  # 步长
    while seq_step >= 1:
        for index in range(0, len(seq), seq_step):  # 决定次数
            for current_index in range(index, 0, -seq_step):
                if seq[current_index] < seq[current_index - seq_step]:
                    seq[current_index], seq[current_index - seq_step] = seq[current_index - seq_step], seq[
                        current_index]
        seq_step -= 1

    return seq


# print(shell_sort(avg))

# 希尔排序平均复杂度O(n^1.5),最佳情况O(n),最差情况O(n^2)
# 空间复杂度O(1)

########################################  计数排序  ####################################
from collections import Counter


def counting_sort(seq):
    from collections import defaultdict
    bucket = defaultdict(int)
    for i in seq:
        bucket[i] += 1

    result = []
    for i in range(min(seq), max(seq) + 1):
        [result.append(i) for _ in range(bucket[i]) if bucket[i]]

    return result


def counting_sort_plus(seq):
    from collections import Counter
    counter = Counter(seq)
    result = []
    for i in range(min(seq), max(seq) + 1):
        if i in counter:
            [result.append(i) for _ in range(counter[i])]

    return result


# print(counting_sort(avg))
# print(counting_sort_plus(avg))


########################################  快速排序  ####################################

def quick_sort(seq):
    if len(seq) == 1:
        return seq

    start, end = 1, len(seq) - 1
    # print(seq,end=' ')
    while start < end:
        while seq[end] >= seq[0] and end > start:  # seq[0] 哨兵
            end -= 1

        while seq[start] <= seq[0] and end > start:
            start += 1

        seq[start], seq[end] = seq[end], seq[start]

    if seq[0] < seq[start]:
        pass
    else:
        seq[0], seq[start] = seq[start], seq[0]
    # print(seq)
    left = quick_sort(seq[:start])  # 左右分治
    right = quick_sort(seq[start:])

    return left + right


# print(quick_sort(avg))


########################################  归并排序  ####################################

def marge_sort(seq):
    if len(seq) == 1:  # 递归最终一定会分成只有一个的情况,直接return
        return seq

    result = []
    middle_index = len(seq) // 2  # 找中点

    right_seq = marge_sort(seq[middle_index:])  # 递归左右分治
    left_seq = marge_sort(seq[:middle_index])
    # print(right_seq, left_seq)

    while right_seq and left_seq:  # 左右合并
        if right_seq[-1] <= left_seq[-1]:
            result.append(right_seq.pop())
        else:
            result.append(left_seq.pop())

    result.reverse()  # reverse是就地修改
    return (right_seq or left_seq) + result  # 左右数量不等会有剩余,并合并结果


# print(marge_sort(avg))


# 归并平均复杂度O(nlogn),最佳情况O(nlogn),最差情况O(nlogn)
# 空间复杂度O(n)

########################################  堆排序  ####################################
# avg = [5, 2, 7, 6, 9, 0, 8, 1, 2, 4, 3, 7]
def heap_adjust(length, index, seq):
    '''
    :param length: 无序区个数
    :param index: 当前要排序的非叶子节点的索引
    :param seq: 数据集
    :return: None
    '''

    # 从顶向下交换
    while 2 ** index <= length:
        left_child_index = 2 * index  # 左孩子节点等于父节点索引乘二,右孩子等于父节点索引乘二加一
        max_child_index = left_child_index  # 假设子树中最大的节点是左孩子节点

        if left_child_index < length and seq[left_child_index + 1] > seq[left_child_index]:
            # 如果存在右孩子节点,并且右孩子比左孩子大那么子树的孩子节点最大值为右孩子
            max_child_index = left_child_index + 1

        if seq[max_child_index] > seq[index]:
            # 判断最大的孩子节点和父节点谁大,把最大的当做父节点
            seq[index], seq[max_child_index] = seq[max_child_index], seq[index]
            index = max_child_index  # 更改目前要排序的索引
        else:
            # 如果这个子树不需要交换,说明已经调整完成,退出循环
            break


def heap_first_adjust(seq):
    # 预先进行一次完整的大顶堆构建
    # 要从最后一个非叶子节点开始,每个节点都要调整才能满足是大顶堆
    length = len(seq)
    for i in range(length // 2, 0, -1):
        # 公式:二叉树的节点总个数整除2,就是有多少个非叶子节点
        # 相当于从最后一个非叶子节点开始,一直到根节点
        heap_adjust(length, i, seq)
    return seq


def heap_sort(seq):
    length = len(seq)  # 无序区个数(待排序元素个数)
    seq = [0] + seq  # 前面补一个没用的,这样用公式好索引
    heap_first_adjust(seq)  # 先构建大顶堆
    while length > 1:
        seq[1], seq[length] = seq[length], seq[1]  # 交换堆顶和最后一个叶子节点,下一次调整堆最后一个叶子节点属于有序区
        length -= 1
        if length == 2 and seq[length] >= seq[length - 1]:
            # 如果无序区就剩两个了,直接比较大小就行了
            break
        heap_adjust(length, 1, seq)  # 因为刚刚最后一个叶子节点和根节点发生了交换,所以要重新对根节点进行排序
    return seq[1:]  # 最后把前面占位的元素剔除


# for版本
def heap_sort_v2(seq):
    seq = [0] + seq
    seq = heap_first_adjust(seq)
    for i in range(len(seq) - 1, 2, -1):
        seq[1], seq[i] = seq[i], seq[1]
        heap_adjust(i - 1, 1, seq)
    if seq[2] >= seq[1]:
        return seq[1:]
    else:
        seq[2], seq[1] = seq[1], seq[2]
        return seq[1:]

# print(heap_sort(avg))
# print(heap_sort_v2(avg))
