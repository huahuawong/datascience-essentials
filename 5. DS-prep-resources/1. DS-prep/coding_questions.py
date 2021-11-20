# This problem was asked by Twitch.

# You are given an m by n matrix with with 0s and 1s, where a 1 represents an obstacle and a 0 represents no obstacle. Determine the number of ways to navigate from the
# top-left corner of the matrix to the bottom right corner given that at any point in time there is only a move down or to the right as long as there is not an obstacle in 
# that spot.

# For example, if the matrix is given by: [[0, 0, 0], [1, 1, 0], [0, 1, 0]] then you should return 1 since there is exactly one path.

# 1. Bear in mind, we can only move right and move down. What we can first do is to first initialze a 2D array with 0s.
# 2. Check the first column to see if there is any obstacle, if no obstacle, assign 1 to that position. Repeat for the first row as well. If an obstacle is found, set the value to 0.
# 3. For the rest of the spaces, fill it with the sum of its top square value and left square value.


def uniquePathsWithObstacles(A):
 
    # create a 2D-matrix and initializing with value 0
    paths = [[0]*len(A[0]) for i in A]
     
    # initializing the left corner if no obstacle there
    if A[0][0] == 0:
        paths[0][0] = 1
     
    # initializing first column of the 2D matrix
    for i in range(1, len(A)):
         
        # If not obstacle
        if A[i][0] == 0:
            paths[i][0] = paths[i-1][0]
             
    # initializing first row of the 2D matrix
    for j in range(1, len(A[0])):
         
        # If not obstacle
        if A[0][j] == 0:
            paths[0][j] = paths[0][j-1]
             
    for i in range(1, len(A)):
        for j in range(1, len(A[0])):
 
            # If current cell is not obstacle
            if A[i][j] == 0:
                paths[i][j] = paths[i-1][j] + paths[i][j-1]
 
    # returning the corner value of the matrix
    return paths[-1][-1]
 
 
# Driver Code
A = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
print(uniquePathsWithObstacles(A))
