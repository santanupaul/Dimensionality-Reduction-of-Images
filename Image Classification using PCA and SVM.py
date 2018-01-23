
# coding: utf-8

# In[ ]:




# In[1]:

import os
import pandas as pd
import numpy as np
from PIL import Image

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

# Import the 3 dimensionality reduction methods
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[2]:

# Change the Current Path
os.chdir('/Users/santanupaul/Documents/Personal/Masters in Analytics/UConn/Study Related/Python/Project/fer2013')

get_ipython().system('pwd')


# In[3]:

# Import the pixel matrix 
df = pd.read_csv('fer2013.csv')

df.head()


# In[5]:

print(df.shape)
df1 = df[(df['emotion'] == 3) | (df['emotion'] == 4)] # 3 happy, and 4 Sad
print(df1.shape)


# In[6]:

a = df1.emotion.value_counts()
print('Happy = ', a[3]/(a[3]+a[4]))
print('Sad = ', a[4]/(a[3]+a[4]))


# In[7]:

df1.Usage.value_counts()
print(df1[df1['Usage'] == 'Training'].groupby(['emotion']).agg({'emotion': 'count'}))
print(df1[df1['Usage'] == 'PublicTest'].groupby(['emotion']).agg({'emotion': 'count'}))
print(df1[df1['Usage'] == 'PrivateTest'].groupby(['emotion']).agg({'emotion': 'count'}))

# All labels are present. We will need to split the data on our own


# In[8]:

# Split the pixel columns to form different fields for each pixel
df1 = pd.concat([df1[['emotion']], df1['pixels'].str.split(" ", expand = True)], axis = 1)
df1.head()

X = df1.iloc[:, 1:].values #Store as numpy array
Y = df1.iloc[:, 0].values

X = X.astype(int)


# In[88]:

X_mean


# In[9]:

# Standardization: Although data is in the same scale (values from 1 to 255) mean = 0 and std = 1 is recommended for most
# machine learning algorithms

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
X_mean = X.mean(0)
X_stddev = X.std(0)

# The classic approach to PCA is to perform the eigendecomposition on the covariance matrix ΣΣ, which is a d×dd×d 
# matrix where each element represents the covariance between two features. The covariance between two features is 
# calculated as follows:
# σjk=1n−1∑Ni=1(xij−x¯j)(xik−x¯k).

# Let's use this approach for PCA (Later we'll use the shortcut)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n', cov_mat)

# Alternative way to calculate covariance
# cov_mat = np.cov(X_std.T)

# Next, we perform an eigendecomposition on the covariance matrix:
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n', eig_vecs)
print('\nEigenvalues \n', eig_vals)


# In[10]:

# Check if eigenvectors are orthonormal and unit vectors

# Orthogonal Property
print(eig_vecs[0, :].dot(eig_vecs[5, :])) #Yields 0
print(eig_vecs[3, :].dot(eig_vecs[12, :])) #Yields 0

# Norm
for i in range(len(eig_vals)):
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(eig_vecs[:, i])) #No Error Thrown


# In[11]:

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
cum_var_exp


# In[12]:

trace1 = go.Scatter(
    x=list(range(784)),
    y= cum_var_exp,
    mode='lines+markers',
    name="'Cumulative Explained Variance'",
    hoverinfo= cum_var_exp,
    line=dict(
        shape='spline',
        color = 'goldenrod'
    )
)
trace2 = go.Scatter(
    x=list(range(784)),
    y= var_exp,
    mode='lines+markers',
    name="'Individual Explained Variance'",
    hoverinfo= var_exp,
    line=dict(
        shape='linear',
        color = 'black'
    )
)
fig = tls.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.5}],
                          print_grid=True)

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2,1,1)
fig.layout.title = 'Explained Variance plots - Full and Zoomed-in'
fig.layout.xaxis = dict(range=[0, 80], title = 'Feature columns')
fig.layout.yaxis = dict(range=[0, 60], title = 'Explained Variance')
fig['data'] += [go.Scatter(x= list(range(784)) , y=cum_var_exp, xaxis='x2', yaxis='y2', name = 'Cumulative Explained Variance')]
fig['data'] += [go.Scatter(x=list(range(784)), y=var_exp, xaxis='x2', yaxis='y2',name = 'Individual Explained Variance')]

# fig['data'] = data
# fig['layout'] = layout
# fig['data'] += data2
# fig['layout'] += layout2
py.iplot(fig, filename='inset example')


# In[13]:

# 107 variables seem to explain 90% of variation (also verified by Sree)
# Let's project the data matrix into new subspace containing 107 dimensions

n_components = 107

matrix_w = eig_pairs[0][1].reshape(len(eig_vecs),1)
for i in range(1, n_components):
    matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(len(eig_vecs),1)))

print(matrix_w) #Contains top 107 Eigen Vectors

# Now project the original X_std into new feature subspace
X_nd = X_std.dot(matrix_w)

print(X_nd.shape)


# In[14]:

eig_vecs_sub = matrix_w.T

n_row = 4
n_col = 7

# Plot the first 8 eignenvalues
plt.figure(figsize=(11,8))

for i in list(range(n_row * n_col)):
#     for offset in [10, 30,0]:
#     plt.subplot(n_row, n_col, i + 1)
    offset =0
    plt.subplot(n_row, n_col, i + 1)
    plt.imshow(eig_vecs_sub[i].reshape(48,48), cmap='jet')
    title_text = 'Eigenvector ' + str(i + 1)
    plt.title(title_text, size=6.5)
    plt.xticks(())
    plt.yticks(())
plt.show()


# In[15]:

# Now let's plot the projected data points for the first two principal components and see how does the labels look like

trace0 = go.Scatter(
    x = X_nd[:,0],
    y = X_nd[:,1],
    name = Y,
    hoveron = Y,
    mode = 'markers',
    text = Y,
    showlegend = False,
    marker = dict(
        size = 8,
        color = Y,
        colorscale ='Jet',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        opacity = 0.8
    )
)
data = [trace0]

layout = go.Layout(
    title= 'Principal Component Analysis (PCA)',
    hovermode= 'closest',
    xaxis= dict(
         title= 'First Principal Component',
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Second Principal Component',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= True
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='styled-scatter')

# There's no observable clusters for the first two PCs


# In[16]:

# Reconstruction of Original Images using reduced dimension
# Refer to amoeba's answer in stackexchange:
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

Xhat = X_nd.dot(eig_vecs_sub)
Xhat *= X_stddev
Xhat += X_mean

img_ind = 628

string = Xhat[img_ind]

arr = np.array(string, dtype=np.uint8)
arr.resize(48, 48)

img = Image.fromarray(arr)
img.show()

string = df1.iloc[img_ind, 1:]

arr = np.array(string, dtype=np.uint8)
arr.resize(48, 48)

img = Image.fromarray(arr)
img.show()

# It appears reconstructed images are not very similar to the original ones so as to discern them categorically. Facial 
# expressions can be subtle and lot more information will be needed to detect them. Sometimes, even naked eyes fail to 
# understand the reconstructed images' emotions. Hence, 90% is not enough information. Let's move to 95% ( > 107 components)

#Before increasing the cutoff, let's run SVM and see how much accuracy do we get


# In[16]:

# Run SVM to see preliminary results
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_nd, Y, random_state = 42,
                                                    test_size = 0.2)

print('Train SVM...')
svc = SVC()
svc.fit(X_train, Y_train)

print('Train Score: \n', svc.score(X_train, Y_train))
print('\n\nTest Score: \n', svc.score(X_test, Y_test))

# Test accuracy is 62%. Not that great.


# In[17]:

# 259 variables seem to explain ~>95% of variation 
# Let's project the data matrix into new subspace containing 259 dimensions

n_components = 259

matrix_w = eig_pairs[0][1].reshape(len(eig_vecs),1)
for i in range(1, n_components):
    matrix_w = np.hstack((matrix_w, eig_pairs[i][1].reshape(len(eig_vecs),1)))

print(matrix_w) #Contains top 107 Eigen Vectors

# Now project the original X_std into new feature subspace
X_nd = X_std.dot(matrix_w)

print(X_nd.shape)

eig_vecs_sub = matrix_w.T


# In[18]:

# Reconstruction of Original Images using reduced dimension
# Refer to amoeba's answer in stackexchange:
# https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com

Xhat = X_nd.dot(eig_vecs_sub)
Xhat *= X_stddev
Xhat += X_mean

img_ind = 532

string = Xhat[img_ind]

arr = np.array(string, dtype=np.uint8)
arr.resize(48, 48)

img = Image.fromarray(arr)
img.show()

string = df1.iloc[img_ind, 1:]

arr = np.array(string, dtype=np.uint8)
arr.resize(48, 48)

img = Image.fromarray(arr)
img.show()

# This is still better. Let's Run SVM and see how much accuracy do we get this time


# In[19]:

# Run SVM to see preliminary results

X_train, X_test, Y_train, Y_test = train_test_split(X_nd, Y, random_state = 42,
                                                    test_size = 0.2)

print('Train SVM...')
svc = SVC()
svc.fit(X_train, Y_train)

print('Train Score: \n', svc.score(X_train, Y_train))
print('\n\nTest Score: \n', svc.score(X_test, Y_test))

# The Test accuracy increased to 65% from 62%

# Just out of curiosity. Let's use all the variables in the model'


# In[20]:

# Run SVM to see preliminary results (Takes a while to run)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 42,
                                                    test_size = 0.2)

print('Train SVM...')
svc = SVC()
svc.fit(X_train, Y_train)

print('Train Score: \n', svc.score(X_train, Y_train))
print('\n\nTest Score: \n', svc.score(X_test, Y_test))

# The Test accuracy drops to 59.7%

# This is because of introduction of noise in the data. Multicollinearity increases. So, we will fix on 259 features.





