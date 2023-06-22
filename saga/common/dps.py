def compute_tl(task, pred_tl):
    if len(task['predecessors']) == 0:
        return 0
    elif any(pred_tl[pred] is None for pred in task['predecessors']):
        return float('inf')
    else:
        return max(pred_tl[pred] + task['cost'][pred] for pred in task['predecessors'])


def compute_bl(task, succ_bl):
    if len(task['successors']) == 0:
        return 0
    elif any(succ_bl[succ] is None for succ in task['successors']):
        return float('inf')
    else:
        return max(succ_bl[succ] + task['cost'][task['processor']] + task['data_transfer'][succ] for succ in task['successors'])


def compute_eft(task, pred_eft):
    if len(task['predecessors']) == 0:
        return task['cost'][task['processor']]
    else:
        return max(pred_eft[pred] + task['cost'][task['processor']] + task['data_transfer'][pred] for pred in task['predecessors'])


def heuristic_dps(tasks, processors):
    n = len(tasks)
    m = len(processors)

    tl = [None] * n
    bl = [None] * n
    priority = [None] * n
    eft = [None] * n
    ready_list = []

    for i in range(n):
        tl[i] = compute_tl(tasks[i], tl)
        bl[i] = compute_bl(tasks[i], bl)
        priority[i] = bl[i] - tl[i]

    task_list = sorted(range(n), key=lambda x: priority[x], reverse=True)

    def update_ready_list():
        nonlocal ready_list
        ready_list = [task for task in ready_list if task in task_list]

    ready_list = task_list.copy()
    schedule = []

    while ready_list:
        task_index = ready_list[0]
        task = tasks[task_index]

        processor = min(range(m), key=lambda x: compute_eft(
            task, eft + [compute_eft(task, eft)]) if x != task['processor'] else float('inf'))

        schedule.append((task['id'], processors[processor]))
        ready_list.remove(task_index)

        for i in range(n):
            if tasks[i]['id'] != task['id'] and tasks[i]['processor'] == task['processor']:
                tl[i] = compute_tl(tasks[i], tl + [tl[task_index]])
                priority[i] = bl[i] - tl[i]

        update_ready_list()

        eft[task_index] = compute_eft(task, eft + [compute_eft(task, eft)])
        bl[task_index] = compute_bl(task, bl + [compute_bl(task, bl)])

    return schedule
