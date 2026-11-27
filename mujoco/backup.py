import math
import numpy as np
"""vector calculation using square root"""

x_distance = 35
y_distance = 25
sqr_values = x_distance**2 + y_distance**2
result_distance = math.sqrt(sqr_values) 
print("resultant distance : ",result_distance)


"""calculating multiple realtion values at same time using linear algebra for finding indidual wheel motor rpm"""

a = np.array([[1,1],[1,-1]])
b = np.array([1,0.2])
x = np.linalg.solve(a,b)
print("Resultant through linear algebra : ",x)

"""calcualting double equation at same time using matrix"""

a = np.array([[2,1],[1,3]])
b = np.array([3,2])
x2 = a @ b 
print("Resultant matrix : ",x2)

"""robot coordinate rotation ,if a robots rotates from its initial position what will be its new coordinate of the tool"""
"""rotation matrix"""
theta=math.radians(90)
a = np.array([[math.cos(theta) , -math.sin(theta)],[math.sin(theta),math.cos(theta)]])
b = np.array([1,0])
reslutant= np.round(a @ b)
print ("resultant position for angle ",theta," is : ",reslutant)

"""Homogeneous transformation matrix"""
theta=math.radians(90)
a = np.array([[math.cos(theta) , -math.sin(theta),2],[math.sin(theta),math.cos(theta),1],[0 , 0 , 1]])
b = np.array([0.966,0.259,1])
reslutant= a @ b
print ("resultant position for angle ",theta," is : ",reslutant)

"""Gaussian Elimination"""
