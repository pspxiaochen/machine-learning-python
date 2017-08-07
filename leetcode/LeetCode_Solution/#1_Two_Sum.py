def twoSum(nums,target):
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            if(nums[i]+nums[j] == target):
                a=[i,j]
    return a
# time complexity:O(n*n)
#############################################

def two_Sum(nums,target):
    dict = {}
    for i in range(len(nums)):
        if (nums[i] in dict):
            return [dict[nums[i]],i]
        else:
            dict[target-nums[i]] = i

# time complexity:O(n)

# test
nums = [2, 7, 11, 15]
target = 9
print(two_Sum(nums,target))