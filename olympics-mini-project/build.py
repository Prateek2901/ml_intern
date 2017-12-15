import pandas as pd
import re


def load_data():
	"""
		Enter your code here
	"""
	df = pd.read_csv("./data/olympics.csv",header=1)
	df = df.rename(columns=lambda x: re.sub('^01 !','Gold',x))
	df = df.rename(columns=lambda x: re.sub('^02 !','Silver',x))
	df = df.rename(columns=lambda x: re.sub('^03 !','Bronze',x))

	country_Name,country_Code =[],[]
	
	for i in df['Unnamed: 0']:
		start = i.find("(")
		end = i.find(")")
		name = i[0:start]
		name = re.sub(r'[^\x00-\x7F]+',' ', name).strip()
		country_Name.append(name)
		country_Code.append(i[start:end+1])

	del df['Unnamed: 0']
	df.insert(0,'country_Name',country_Name)
	df.insert(1,'country_Code',country_Code)
	df = df.set_index('country_Name')
	df.drop(df.tail(1).index,inplace=True)
	
	return df

def first_country(df):
	return df.iloc[0]


def gold_medal(df):
	return df.loc[df['Gold'].argmax()].name


def biggest_difference_in_gold_medal(df):
	return df.loc[(df['Gold'] - df['Gold.1']).abs().idxmax()].name

def get_points(df):
	points = []
	for index, row in df.iterrows():
		pt = row['Gold.2']*3+row['Silver.2']*2+row['Bronze.2']
		points.append(pt)
		
	df['Points']=points
	return df

def k_means(df):
	data = df[['# Games','Points']]
	from sklearn.cluster import KMeans
	k = 3
	kmeans =  KMeans(n_clusters=3).fit(data)
	return (k,kmeans.cluster_centers_)
