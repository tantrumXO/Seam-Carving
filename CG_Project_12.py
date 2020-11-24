#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
from matplotlib.pyplot import imshow
from PIL import Image


# In[31]:


# Opening Image

img = Image.open('image.jpg')
print(img.mode)
print(img.size)
imshow(np.asarray(img))


# In[32]:


img_array = np.array(img)
print(img_array.shape)


# In[33]:


# Energy Calculation (M1.1)

# energy = np.zeros((img.size[1], img.size[0]))
energy = img_array.copy()

for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        neighbors = {}
        
        # Edge Cases
        if i == 0:
            neighbors['up'] = img_array[i][j]
        
        if j == 0:
            neighbors['left'] = img_array[i][j]
        
        if i == img_array.shape[0] - 1:
            neighbors['down'] = img_array[i][j]
        
        if j == img_array.shape[1] - 1:
            neighbors['right'] = img_array[i][j]
        
        # General Case
        if 'up' not in neighbors:
            neighbors['up'] = img_array[i-1][j]
        
        if 'down' not in neighbors:
            neighbors['down'] = img_array[i+1][j]
        
        if 'left' not in neighbors:
            neighbors['left'] = img_array[i][j-1]
        
        if 'right' not in neighbors:
            neighbors['right'] = img_array[i][j+1]
        
        dx_r = int(neighbors['left'][0]) - int(neighbors['right'][0])
        dx_g = int(neighbors['left'][1]) - int(neighbors['right'][1])
        dx_b = int(neighbors['left'][2]) - int(neighbors['right'][2])
        dx = (dx_r ** 2) + (dx_g ** 2) + (dx_b ** 2)
        
        dy_r = int(neighbors['up'][0]) - int(neighbors['down'][0])
        dy_g = int(neighbors['up'][1]) - int(neighbors['down'][1])
        dy_b = int(neighbors['up'][2]) - int(neighbors['down'][2])
        dy = (dy_r ** 2) + (dy_g ** 2) + (dy_b ** 2)
        
        e = np.sqrt(dx + dy)
        energy[i][j] = int(e)

print(energy.shape)
print(energy)


# In[34]:


# Displaying Energy (M1.2)

energy_img = Image.fromarray(energy)
# energy_img.show()
imshow(np.asarray(energy_img))


# In[35]:


# Computing Seams (M2)

dp_vertical = np.zeros((energy.shape[0], energy.shape[1]))

for j in range(dp_vertical.shape[1]):
    dp_vertical[0][j] = energy[0][j][0]

for i in range(1, dp_vertical.shape[0]):
    for j in range(dp_vertical.shape[1]):
        if j == 0:
            dp_vertical[i][j] = int(energy[i][j][0]) + min(energy[i-1][j][0], energy[i-1][j+1][0])
        elif j == dp_vertical.shape[1] - 1:
            dp_vertical[i][j] = int(energy[i][j][0]) + min(energy[i-1][j-1][0], energy[i-1][j][0])
        else:
            dp_vertical[i][j] = int(energy[i][j][0]) + min(energy[i-1][j-1][0], energy[i-1][j][0], energy[i-1][j+1][0])

pd.DataFrame(dp_vertical)


# In[36]:


seams = []

def vertical_seam(dp_vertical):
    seam = []
    minimum = min(dp_vertical[-1])
    print("minimum in last row: " + str(minimum))
    index = np.where(dp_vertical[-1] == minimum)
    print("index: " + "j = " + str(index[0][0]))
    i = len(dp_vertical)-1
    j = index[0][0]
    seam.append([i, j])

    for row in range(len(dp_vertical) - 1, 0, -1):
        if j == 0:
            up = dp_vertical[row-1][j]
            up_right = dp_vertical[row-1][j+1]
            if up_right < up:
                i = row-1
                j = j+1
                seam.append([i, j])
            else:
                i = row-1
                j = j
                seam.append([i, j])
    
        elif j == len(dp_vertical[row]) - 1:
            up = dp_vertical[row-1][j]
            up_left = dp_vertical[row-1][j-1]
            if up_left < up:
                i = row-1
                j = j-1
                seam.append([i, j])
            else:
                i = row-1
                j = j
                seam.append([i, j])
    
        else:
            up = dp_vertical[row-1][j]
            up_left = dp_vertical[row-1][j-1]
            up_right = dp_vertical[row-1][j+1]
            if up_left <= up and up_left <= up_right:
                i = row-1
                j = j-1
                seam.append([i, j])
            elif up_right <= up and up_right <= up_left:
                i = row-1
                j = j+1
                seam.append([i, j])
            else:
                i = row-1
                j = j
                seam.append([i, j])
    return seam

seams.append(vertical_seam(dp_vertical))
print(seams)


# In[37]:


for coord in seams[0]:
    energy[coord[0]][coord[1]] = [255, 0 ,0]

energy_img = Image.fromarray(energy)
imshow(np.asarray(energy_img))
energy_img.show()


# In[ ]:




