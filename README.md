# Ex-06-Feature-Transformation

AIM:

To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM:

STEP 1

Read the given Data

STEP 2

Clean the Data Set using Data Cleaning Process

STEP 3

Apply Feature Transformation techniques to all the features of the data set

STEP 4

Save the data to the file

CODE

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.isnull().sum()

df.describe()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```

OUTPUT:

DATASET

![image](https://user-images.githubusercontent.com/118626456/234180066-c3234f9a-06f1-414f-9aa1-3c1e0e353d2c.png)

ISNULL:

![image](https://user-images.githubusercontent.com/118626456/234180132-d6908b85-c6d5-4a7b-a6fb-55857ba61a3b.png)

INFO:

![image](https://user-images.githubusercontent.com/118626456/234180191-f6c9c81b-da06-4929-8a02-7be7b6cbf827.png)

DESCRIBE:

![image](https://user-images.githubusercontent.com/118626456/234180247-a05efa44-0e73-4be4-bee9-44a348c6f853.png)

HIGHLY POSITIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180305-cd8593f1-2092-4854-af8a-f699d63363c7.png)

HIGHLY NEGATIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180339-e196babd-fd3b-4e5b-a49d-e49169cfdf0d.png)

MODERATE POSITIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180384-fc049295-2782-4ef2-a03a-3ee6f1fd2a46.png)

MODERATE NEGATIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180434-7b5ab6c2-82fe-4306-b992-60163d43f383.png)

LOG OF MODERATE POSITIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180478-8fdabc0b-4686-45fe-b551-8db21d5fba50.png)

LOG OF HIGHLY POSITIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180524-ac3f8e6f-1262-4486-82b0-7227c83d2047.png)

RECIPROCAL OF HIGHLY POSITIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180565-8c76fca7-4b3a-40dc-80ce-fd4d87e4229a.png)

SQUARE ROOT TRANSFORMATION:

![image](https://user-images.githubusercontent.com/118626456/234180616-7196e862-928a-4e37-bfe1-377e24b1190c.png)

POWER TRANSFORMATION OF MODERATE NEGATIVE SKEW:

![image](https://user-images.githubusercontent.com/118626456/234180671-8bfced4e-0091-43fe-a0b6-7c28011f2acf.png)

QUANTILE TRANSFORMATION:

![image](https://user-images.githubusercontent.com/118626456/234180719-e0753feb-1ec9-4d65-8e7b-f023ce97b03b.png)

RESULT:

Thus Feature transformation is performed and executed successfully for the given dataset
