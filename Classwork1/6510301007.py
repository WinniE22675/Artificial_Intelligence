import numpy as np 
import matplotlib.pyplot as plt

x = np.array([29,28,34,31,25])  
y = np.array([77,62,93,84,59]) 
n = np.size(x) 

x_mean = np.mean(x)
y_mean = np.mean(y)
x_mean,y_mean   

Sxy = np.sum(x*y)- n*x_mean*y_mean  # The sum of products of x and y
Sxx = np.sum(x*x)-n*x_mean*x_mean   # The sum of squares of y
  
b1 = Sxy/Sxx                        # a : ความชัน (slope)
b0 = y_mean-b1*x_mean               # b : จุดตัด (bias/y-intercept) 
print(f'slope a is {b1:.2f}') 
print(f'y-intercept b is {b0:.2f}') 
  
plt.scatter(x,y) 
plt.xlabel('Independent variable X') 
plt.ylabel('Dependent variable y') 

y_pred = b1 * x + b0 

# Adding labels and title
plt.scatter(x, y, color = 'red') 
plt.plot(x, y_pred, color = 'green') 
plt.xlabel('X') 
plt.ylabel('y') 
plt.title('Regression Equation')

# Display the plot
plt.show()