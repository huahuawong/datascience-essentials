#1 Implement the function unique_in_order which takes as argument a sequence and returns a list of items without any elements with the same value next
# to each other and preserving the original order of elements.

def unique_in_order(iterable):
    res = []
    for item in iterable:
        if len(res) == 0 or item != res[-1]:
            res.append(item)
    return res
  
# unique_in_order('AAAABBBCCDAABBB') == ['A', 'B', 'C', 'D', 'A', 'B']
 
  
#2 Count the number of Duplicates
# Write a function that will return the count of distinct case-insensitive alphabetic characters and numeric digits that occur more than once in the input string. 
# The input string can be assumed to contain only alphabets (both uppercase and lowercase) and numeric digits.

def duplicate_count(text):
    text = text.lower()
    duplicates = []
    for i in text:
        if text.count(i) > 1 and i not in duplicates:
            duplicates.append(i)    
    return len(duplicates)
  
def duplicate_count(text):
    seen = set()
    dupes = set()
    for char in text:
        char = char.lower()
        if char in seen:
            dupes.add(char)
        seen.add(char)
    return len(dupes)
  
# "aabBcde" -> 2 # 'a' occurs twice and 'b' twice (`b` and `B`)
