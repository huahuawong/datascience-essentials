# This problem is part of your free preview of Interview Query Premium.
# This question was asked by: Splunk

# We're given a string of integers that represent page numbers.
# Write a function to return the last page number in the string. If the string of integers is not in correct page order, return the last number in order.

# 1. First, initialize variable 'pos' and 'page' with zero
# 2. Create a loop, basically it'll check if the next character matches 'page_str', which is the next 'page', 'page' is constantly updated if the criteria is met, i.e. it is 
# indeed the next variable.
# 3. We have to update 'pos' as well with the length of the 'page_str', we can't use increment by 1, since it can reach double/ triple digits
# 4. If it is not the next number, we break the loop, and return the 'page' variable.

def last_page_number(string):
    pos = 0
    page = 0
    while pos < len(string):
        page_str=str(page+1)
        if string[pos:pos+len(page_str)]==page_str:
            pos+=len(page_str)
            page+=1
        else:
            break
    return page
  
