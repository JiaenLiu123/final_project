def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Split the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Recursively sort each half
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Merge the sorted halves
    sorted_arr = []
    i = 0
    j = 0
    while i < len(left_half) and j < len(right_half):
        if left_half[i] < right_half[j]:
            sorted_arr.append(left_half[i])
            i += 1
        else:
            sorted_arr.append(right_half[j])
            j += 1
    
    # Add any remaining elements from either half
    while i < len(left_half):
        sorted_arr.append(left_half[i])
        i += 1
    while j < len(right_half):
        sorted_arr.append(right_half[j])
        j += 1
    
    return sorted_arr


arr = [3, 7, 2, 1, 8, 4, 5, 6]
sorted_arr = merge_sort(arr)
print(sorted_arr)  # Outputs [1, 2, 3, 4, 5, 6, 7, 8]
