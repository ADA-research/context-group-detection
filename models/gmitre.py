def compute_mitre(a, b):
    """
    compute mitre 
    more details: https://aclanthology.org/M95-1005.pdf
    args:
      a,b: list of groups; e.g. a=[[1,2],[3],[4]], b=[[1,2,3],[4]]
    Return: 
      mitreLoss a_b
      
    """
    total_m = 0  # total missing links
    total_c = 0  # total correct links
    for group_a in a:
        pa = 0  # partitions of group_a in b
        part_group = []  # partition group
        size_a = len(group_a)  # size of group a
        for element in group_a:
            for group_b in b:
                if element in group_b:
                    if part_group == group_b:
                        continue
                    else:
                        part_group = group_b
                        pa += 1
        total_c += size_a - 1
        total_m += pa - 1

    return (total_c - total_m) / total_c


def create_counterPart(a):
    """
    add fake counterparts for each agent
    args:
      a: list of groups; e.g. a=[[0,1],[2],[3,4]]
    """
    a_p = []
    for group in a:
        if len(group) == 1:  # singleton
            element = group[0]
            element_counter = -(element + 1)  # assume element is non-negative
            new_group = [element, element_counter]
            a_p.append(new_group)
        else:
            a_p.append(group)
            for element in group:
                element_counter = -(element + 1)
                a_p.append([element_counter])
    return a_p


def compute_groupMitre(target, predict):
    """
    compute group mitre
    args: 
      target,predict: list of groups; [[0,1],[2],[3,4]]
    return: recall, precision, F1
    """
    # create fake counter agents
    target_p = create_counterPart(target)
    predict_p = create_counterPart(predict)
    recall = compute_mitre(target_p, predict_p)
    precision = compute_mitre(predict_p, target_p)
    if recall == 0 or precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return recall, precision, f1
