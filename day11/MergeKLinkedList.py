def create_node(val):
    return {'val': val, 'next': None}

def build_linked_list(arr):
    if not arr:
        return None
    head = create_node(arr[0])
    current = head
    for val in arr[1:]:
        current['next'] = create_node(val)
        current = current['next']
    return head

def merge_two_lists(l1, l2):
    dummy = create_node(0)
    tail = dummy
    while l1 and l2:
        if l1['val'] < l2['val']:
            tail['next'] = l1
            l1 = l1['next']
        else:
            tail['next'] = l2
            l2 = l2['next']
        tail = tail['next']
    tail['next'] = l1 if l1 else l2
    return dummy['next']

def merge_k_lists(lists):
    if not lists:
        return None
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i+1] if i+1 < len(lists) else None
            merged.append(merge_two_lists(l1, l2))
        lists = merged
    return lists[0]

def linked_list_to_array(node):
    result = []
    while node:
        result.append(node['val'])
        node = node['next']
    return result
lists = [
    build_linked_list([1, 4, 5]),
    build_linked_list([1, 3, 4]),
    build_linked_list([2, 6])
]

merged = merge_k_lists(lists)
print(linked_list_to_array(merged))  # Output: [1, 1, 2, 3, 4, 4, 5, 6]