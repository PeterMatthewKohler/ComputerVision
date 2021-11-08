#Homework 0
#By: Peter Kohler

import math
import numpy as np

def gaussian(x, y, sigma):      #2-D isotropic Gaussian Formula and Function Declaration

    return ( (1/ (2*np.pi*(sigma**2)) ) * np.exp(-((x**2) + (y**2))/(2*sigma**2) )  )



if __name__ == '__main__':
    #Python Programming
    #Section 1
    print("Section 1")
    #-------------------------------------------------------------------------
    # Question 1
    A = np.array([[4, -2], [1, 1]])
    # Part a.)  Determinant of A
    print("1.) \na) Determinant of A = ", np.linalg.det(A))
    # Part b.) Trace of A
    print("b) Trace of A = ", np.trace(A))
    # Part c.) Inverse of A
    print("c) Inverse of A = \n", np.linalg.inv(A))
    # Part d. and e.) Eigenvalues and Eigenvectors of A
    evalue, evect = np.linalg.eig(A)
    print("d) Eigenvector of A:\n", evect)
    print("e) Eigenvalues of A:\n", evalue)

    #Question 2
    B = np.array([[3, 4],[5, -1]])
    # Part a.) Compute (AB)^T
    result1 = np.matmul(A, B)
    result1 = np.matrix.transpose(result1)
    print("\n2.)\na.) Compute (AB)^T\n", result1)
    # Part b.) Compute (B^T)(A^T)
    result1 = np.matrix.transpose(A)
    result2 = np.matrix.transpose(B)
    result3 = np.matmul(result2, result1)
    print("b.) Compute (B^T)*(A^T)\n", result3)

    #Question 3
    x = np.array([1, 2, 3])
    y = np.array([-1, 2, -3])
    # Part a.) Compute x dot y
    print("3.)\nPart a.) x dot y = \n", np.dot(x, y))
    # Part b.) Compute x cross y
    print("Part b.) x cross y = \n", np.cross(x, y))
    print("------------------------------------------------------------")
    #----------------------------------------------------------------------------------

    #Section 2
    mu, sigma = 0, 1.000 #mean and std. deviation values
    total = 0

    result = [[0] * 9 for i in range(9)]
    for i in range(9):
        for j in range(9):
            result[i][j] = gaussian(i-4, j-4, sigma)
            total += gaussian(i-4, j-4, sigma)
    print("\nSection2",
          "\n\nThe sum of this array is = ", total)
    print("\nThe values in this 9x9 array are: ")
    for i in range(9):
        for j in range(9):
            print( result[i][j], end = ", ")
            if (j == 8):
                print("\n")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/