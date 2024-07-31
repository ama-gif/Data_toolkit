#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# QUS 1 -  Demonstrate three different methods for creating identical 2D arrays in NumPy. Provide the code for each 
# method and the final output after each method

# In[7]:


array1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) ## method 1 Using np.array()
print(array1)


# In[8]:


array2 = np.zeros((3, 3), dtype=int)
array2[0] = [1, 2, 3]
array2[1] = [4, 5, 6]                                 ## Using np.zeros() and then filling it
array2[2] = [7, 8, 9]
print(array2)


# QUS 2 -  Using the Numpy function, generate an array of 100 evenly spaced numbers between 1 and 10 and 
# Reshape that 1D array into a 2D array

# In[13]:


numbers = np.linspace(1, 10, 100)
numbers_1d = numbers.reshape(-1)

print(numbers_1d)


# In[15]:


numbers = np.linspace(1, 10, 100)
numbers_2d = numbers.reshape(10, 10)
print(numbers_2d)


# QUS 3 - Explain the following terms
#  The difference in np.array, np.asarray and np.asanyarray
#  The difference between Deep copy and shallow copy

# np.array - The np.array() in Python is used to convert a list, tuple, etc. into a Numpy array.
# 
# np.sarray - The numpy.asarray()function is used when we want to convert the input to an array.
# Input can be lists, lists of tuples, tuples,tuples of tuples, tuples of lists and arrays.
# 
# np.asanyarray - Similar to np.asarray, but it passes through subclasses of ndarray.
#     This means that if the input is an instance of a subclass of ndarray, it won't be converted to a base ndarray.
#     
# Deep copy - Deep copy stores copies of the object’s value.
# Deep copy stores the copy of the original object and recursively copies the objects as well.
# 
# Shallow copy - Shallow Copy stores the references of objects to the original memory address.   
# Shallow Copy stores the copy of the original object and points the references to the objects.

# QUS 4 - Generate a 3x3 array with random floating-point numbers between 5 and 20. Then, round each number in 
# the array to 2 decimal places

# In[11]:


random_array = np.random.uniform(5, 20, (3, 3))
print(random_array)


# In[12]:


rounded_array = np.round(random_array, 2)
print(rounded_array)


# QUS 5 - Create a NumPy array with random integers between 1 and 10 of shape (5, 6). After creating the array 
# perform the following operations:
#  a)Extract all even integers from array.
#  b)Extract all odd integers from array

# In[17]:


random_array = np.random.randint(1, 11, size=(5, 6))
print(random_array)


# In[18]:


even_integers = random_array[random_array % 2 == 0]
print(even_integers)


# In[19]:


odd_integers = random_array[random_array % 2 != 0]
print(odd_integers)


# QUS 6 - Create a 3D NumPy array of shape (3, 3, 3) containing random integers between 1 and 10. Perform the 
# following operations:
#  a) Find the indices of the maximum values along each depth level (third axis).
#  b) Perform element-wise multiplication of between both arraY

# In[20]:


array_3d = np.random.randint(1, 11, (3, 3, 3))
print(array_3d)


# In[21]:


max_indices = np.argmax(array_3d, axis=2)
print(max_indices)


# In[23]:


multiplied_array = array_3d * 2
print(multiplied_array)


# QUS 7 -  Clean and transform the 'Phone' column in the sample dataset to remove non-numeric characters and 
# convert it to a numeric data type. Also display the table attributes and data types of each column

# In[29]:


data = {
    'Name': ['Aman', 'Aryan', 'Harshit'],
    'Phone': ['123-456-7890', '(234)a567-8901', '+345 678 9012']
}


# In[30]:


df = pd.DataFrame(data)
print(df)


# In[31]:


df['Phone'] = df['Phone'].replace({'\D': ''}, regex=True)
df['Phone'] = pd.to_numeric(df['Phone'], errors='coerce')


# In[32]:


print(df)
print(df.dtypes)


# QUS 8 -  Perform the following tasks using people dataset:
#  a) Read the 'data.csv' file using pandas, skipping the first 50 rows.
#  b) Only read the columns: 'Last Name', ‘Gender’,’Email’,‘Phone’ and ‘Salary’ from the file.
#  c) Display the first 10 rows of the filtered dataset.
#  d) Extract the ‘Salary’' column as a Series and display its last 5 values

# In[42]:


df = pd.read_csv(r"C:\Users\amanp\Downloads\People Data (1).csv"  )


# In[43]:


df


# In[44]:


df_filtered = df[['Last Name', 'Gender', 'Email', 'Phone', 'Salary']]


# In[45]:


print("First 10 rows of the filtered dataset:")
print(df_filtered.head(10))


# In[46]:


salary_series = df_filtered['Salary']
print("\nLast 5 values of the 'Salary' column:")
print(salary_series.tail(5))


# QUS 9 - Filter and select rows from the People_Dataset, where the “Last Name' column contains the name 'Duke',  
# 'Gender' column contains the word Female and ‘Salary’ should be less than 85000

# In[47]:


filtered_df = df[
    (df['Last Name'].str.contains('Duke', na=False)) & 
    (df['Gender'].str.contains('Female', na=False)) & 
    (df['Salary'] < 8500)
]
print(filtered_df)


# QUS 10 - Create a 7*5 Dataframe in Pandas using a series generated from 35 random integers between 1 to 6?
# 

# In[48]:


random_numbers = np.random.randint(1, 7, size=35)
data = random_numbers.reshape(7, 5)
df = pd.DataFrame(data)

print(df)


# QUS 11 - Practice Questions
#  Create two different Series, each of length 50, with the following criteria:
#  a) The first Series should contain random numbers ranging from 10 to 50.
#  b) The second Series should contain random numbers ranging from 100 to 1000.
#  c) Create a DataFrame by joining these Series by column, and, change the names of the columns to 'col1', 'col2', 
# etc

# In[49]:


series1 = pd.Series(np.random.randint(10, 51, size=50))
# b) Create the second Series with random numbers from 100 to 1000
series2 = pd.Series(np.random.randint(100, 1001, size=50))
# c) Create a DataFrame by joining the Series and rename columns
df = pd.concat([series1, series2], axis=1)
df.columns = ['col1', 'col2']

print(df)


#  QUS 12 - Perform the following operations using people data set:
#  a) Delete the 'Email', 'Phone', and 'Date of birth' columns from the dataset.
#  b) Delete the rows containing any missing values.
#  d) Print the final output als

# In[54]:


df = pd.read_csv(r"C:\Users\amanp\Downloads\People Data (1).csv"  )


# In[55]:


print("Column names in the dataset:", df.columns)
df = df.drop(columns=['Email', 'Phone', 'Date of birth'])
df = df.dropna()
print(df)


# QUS 13 -  Create two NumPy arrays, x and y, each containing 100 random float values between 0 and 1. Perform the 
# following tasks using Matplotlib and NumPy:
#  a) Create a scatter plot using x and y, setting the color of the points to red and the marker style to 'o'.
#  b) Add a horizontal line at y = 0.5 using a dashed line style and label it as 'y = 0.5'.
#  c) Add a vertical line at x = 0.5 using a dotted line style and label it as 'x = 0.5'.
#  d) Label the x-axis as 'X-axis' and the y-axis as 'Y-axis'.
#  e) Set the title of the plot as 'Advanced Scatter Plot of Random Values'.
#  f) Display a legend for the scatter plot, the horizontal line, and the vertical line

# In[59]:


x = np.random.rand(100)
y = np.random.rand(100)


# In[60]:


plt.scatter(x, y, color='red', marker='o', label='Data Points')


# In[61]:


plt.axhline(y=0.5, color='blue', linestyle='--', label='y = 0.5')


# In[62]:


plt.axvline(x=0.5, color='green', linestyle=':', label='x = 0.5')



# In[63]:


plt.xlabel('X-axis')
plt.ylabel('Y-axis')


# In[64]:


plt.title('Advanced Scatter Plot of Random Values')


# In[65]:


plt.legend()
plt.show()


# QUS 14 - Create a time-series dataset in a Pandas DataFrame with columns: 'Date', 'Temperature', 'Humidity' and 
# Perform the following tasks using Matplotlib:
#  right y-axis for 'Humidity').
#  b) Label the x-axis as 'Date'.
#  a) Plot the 'Temperature' and 'Humidity' on the same plot with different y-axes (left y-axis for 'Temperature' and 
# c) Set the title of the plot as 'Temperature and Humidity Over Time'

# In[66]:


# Create a date range
dates = pd.date_range(start='2023-01-01', end='2023-12-31')

# Create random data for temperature and humidity
temperature = np.random.randint(15, 35, size=len(dates))
humidity = np.random.randint(40, 90, size=len(dates))

# Create DataFrame
data = pd.DataFrame({'Date': dates, 'Temperature': temperature, 'Humidity': humidity})
# Create the plot
fig, ax1 = plt.subplots()


# In[70]:


# Plot temperature on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (C)', color=color)
ax1.plot(data['Date'], data['Temperature'], color=color)
ax1.tick_params(axis='y', labelcolor=color)


# In[71]:


# Create the second y-axis for humidity
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Humidity (%)', color=color)
ax2.plot(data['Date'], data['Humidity'], color=color)
ax2.tick_params(axis='y', labelcolor=color)


# In[72]:


plt.title('Temperature and Humidity Over Time')

plt.show()


#  QUS 15 -Create a NumPy array data containing 1000 samples from a normal distribution. Perform the following 
# tasks using Matplotlib:
#  a) Plot a histogram of the data with 30 bins.
#  b) Overlay a line plot representing the normal distribution's probability density function (PDF).
#  c) Label the x-axis as 'Value' and the y-axis as 'Frequency/Probability'.
#  d) Set the title of the plot as 'Histogram with PDF Overlay'

# In[104]:


data = np.random.randn(1000)
mean, std = np.mean(data), np.std(data)


# In[105]:


# Create the histogram
plt.hist(data, bins=30, density=True, alpha=0.6, label='Histogram')


# In[75]:


from scipy.stats import norm


# In[76]:


x = np.linspace(min(data), max(data), 100)

# Plot the PDF
plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label='PDF')


# In[77]:


# Set labels and title
plt.xlabel('Value')
plt.ylabel('Frequency/Probability')
plt.title('Histogram with PDF Overlay')

plt.legend()

plt.show()


# QUS 16- Set the title of the plot as 'Histogram with PDF Overlay'.

#  QUS 17 -Create a Seaborn scatter plot of two random arrays, color points based on their position relative to the 
# origin (quadrants), add a legend, label the axes, and set the title as 'Quadrant-wise Scatter Plot'

# In[88]:


x = np.random.randn(100)
y = np.random.randn(100)

# Create a DataFrame for easier manipulation
data = pd.DataFrame({'x': x, 'y': y})

# Determine quadrant based on x and y values
def get_quadrant(row):
    if row['x'] > 0 and row['y'] > 0:
        return 'Top Right'
    elif row['x'] < 0 and row['y'] > 0:
        return 'Top Left'
    elif row['x'] < 0 and row['y'] < 0:
        return 'Bottom Left'
    else:
        return 'Bottom Right'

data['quadrant'] = data.apply(get_quadrant, axis=1)

# Create the scatter plot
sns.scatterplot(data=data, x='x', y='y', hue='quadrant', palette='bright')
# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quadrant-wise Scatter Plot')

plt.show()


# QUS 18 - With Bokeh, plot a line chart of a sine wave function, add grid lines, label the axes, and set the title as 'Sine 
# Wave Function

# In[91]:


from bokeh.plotting import figure, show
from bokeh.models import Grid, Label, Title


# In[92]:


x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(x)

p = figure(title="Sine Wave Function", x_axis_label='x', y_axis_label='y')

p.line(x, y, line_width=2)

p.add_layout(Grid(dimension=0, ticker=p.xaxis[0].ticker))
p.add_layout(Grid(dimension=1, ticker=p.yaxis[0].ticker))

show(p)


# In[ ]:


QUS 19- Using Bokeh, generate a bar chart of randomly generated categorical data, color bars based on their 
values, add hover tooltips to display exact values, label the axes, and set the title as 'Random Categorical 
Bar Chart


# In[96]:


from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool


# In[97]:


categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(1, 101, size=5)

# Create ColumnDataSource
source = ColumnDataSource(data={'categories': categories, 'values': values})

# Create the figure
p = figure(title="Random Categorical Bar Chart", x_axis_label='Category', y_axis_label='Value')

# Add bar chart
p.vbar(x='categories', top='values', width=0.5, source=source, color='values', line_color='black')

# Add hover tool
hover = HoverTool(tooltips=[('Category', '@categories'), ('Value', '@values')])
p.add_tools(hover)
show(p)


# QUS20 - Using Plotly, create a basic line plot of a randomly generated dataset, label the axes, and set the title as 
# 'Simple Line Plot

# In[99]:


import plotly.express as px


# In[102]:


x = np.linspace(0, 10, 100)
y = np.random.randn(100)

# Create the figure
fig = px.line(x=x, y=y, title='Simple Line Plot')

# Update axis labels
fig.update_layout(xaxis_title='X', yaxis_title='Y')

fig.show()



# In[101]:


x = np.linspace(0, 10, 100)
y = np.random.randn(100)

# Create the figure
fig = px.line(x=x, y=y, title='Simple Line Plot')

# Update axis labels
fig.update_layout(xaxis_title='X', yaxis_title='Y')

fig.show()


# QUS 21 -  Using Plotly, create an interactive pie chart of randomly generated data, add labels and percentages, set 
# the title as 'Interactive Pie Chart'

# In[103]:


labels = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(1, 101, size=5)

# Create the figure
fig = px.pie(values=values, names=labels, title='Interactive Pie Chart')

fig.show()


# In[ ]:




