from __future__ import division
from apyori import apriori
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import random
import operator

#for every basket, convert list of lists to list of strings
def flat_baskets (basket):
	big_list = []
	for prod in basket:
		flat_list = []
		for sublist in prod:
			for item in sublist.split(","):
				flat_list.append(item) 
		big_list.append(flat_list)
	return big_list

#for every product in basket assign its subclass vector
#for every basket aggregate all product subclass vectors to one
def agg_vec (baskets, products, category):
	agg_list = []
	for prod in baskets:
		vector_list = []
		for p in prod:	
			if p in products:
				index = products.index(p)
				vector_list.append(category[index])		#get subclass vector for every product

		agg_vector = [sum(elts) for elts in zip(*vector_list)]		#aggregate subclass vectors for every basket
		agg_list.append(agg_vector)

	return agg_list

#assign to every product its class or subclass
def assign (baskets, products, category):
	agg_list = []	
	for prod in baskets:
		sublist = []
		for p in prod.split(","):
			if p in products:
				index = products.index(p)
				sublist.append(category[index]) 	#category: classes or subclasses
		agg_list.append(sublist)
	return agg_list

def safe_div(x,y):
	if y==0:
		return 0
	else:
		return x/y	

#two normalizations of customer vector
def normalization(cms):
	cms1_list = []
	cms2_list = []
	for cust in cms:                         #for every customer
		cms1 = [x/sum(cust) for x in cust]   #divide subclass vector of customer with customer's total spending in subclasses
		cms1 = [round(x,4) for x in cms1]
		cms1_list.append(cms1)
	
	agg_sub_vector = [sum(elts) for elts in zip(*cms1_list)]  #aggregate all normalized cms vectors to one
	agg_sub_vector = [round(x,4) for x in agg_sub_vector]


	for cust in cms1_list:  							#for every customer
		div = [x/len(cms) for x in agg_sub_vector]  	#divide aggregated normalized vector by number of customers (lets call it ratio 1)
		div = [round(x,4) for x in div]
		cms2 = [safe_div(x,y) for x,y in zip(cust,div)]	#divide normalized cms by ratio 1
		cms2 = [round(x,4) for x in cms2]
		cms2_list.append(cms2)

	return cms2_list

#produce association rules for 1-1 classes/subclasses
def association_rules(transactions,sup,conf):
	results = list(apriori(transactions, min_support = sup, min_confidence = conf))  #apriori algorithm
	first_item = []
	second_item = []
	for relation in results:
		if len(relation[0]) == 2:  #we keep only 1-1 relations
			for ordered in relation[2]:
				for first_relation in ordered[0]:
					first_item.append(first_relation)  #antecedent relation
				for second_relation in ordered[1]:
					second_item.append(second_relation)  #consequent relation

	rules_df = pd.DataFrame({'first':first_item, 'second':second_item}, columns=['first','second'])
	return rules_df

#check if two classes/subclasses are associated
def associated(rules, this, s):
	exists = 0
	first_relation = rules['first'].tolist()	#get column of association's first element
	second_relation = rules['second'].tolist()	#get column of association's second element
	for r in first_relation:
		if r == this:
			ind = first_relation.index(r)		
			if second_relation[ind] == s:		#if an association between current and examined class/subclass exists
				exists = 1						#return 1

	return exists

#create product vectors
def product_vector(subclasses,products,classes,rules_sub,rules_cl):
	unique_subclasses = list(set(subclasses))
	unique_subclasses.sort()

	product_vectors = []
	for p in products:
		prod_vec = []
		ind = products.index(p)
		this_sub = subclasses[ind] 		#get subclass of current product
		this_cl = classes[ind] 			#get class of current product
		for s in unique_subclasses: 	#examine all subclasses
			ind_sub = subclasses.index(s)
			c = classes[ind_sub]  		#get class of currently examined subclass
			if this_sub == s: 			#within same subclass
				prod_vec.append(1)
			elif associated(rules_sub,this_sub,s) == 1:		#within associated subclass
				prod_vec.append(1)
			elif this_cl == c:								#within same class
				prod_vec.append(0.5)
			elif associated(rules_cl, this_cl, c) == 1:		#within subclass of associated class
				prod_vec.append(0.25)
			else:
				prod_vec.append(0)

		product_vectors.append(prod_vec)

	return product_vectors

#score every product and return recommendations for every customer according to his basket
def score(agg_baskets, cms_norm, products, product_vectors, subclasses, classes):
	rec_list = []
	count_id = []
	out_count = 1
	rn = 1 	#modulation factor
	for b in agg_baskets:
		score_list = []
		prod_list = []
		cust_index = out_count-1
		cust_vector = cms_norm[cust_index]	#get customer vector
		for p in products:
			if p not in b:	#examine products that customer has not bought
				prod_index = products.index(p)
				prod_vector = product_vectors[prod_index]	#get product vector
				inner = np.inner(cust_vector,prod_vector)	#inner product
				cm = 0
				for x in cust_vector:		#sum of vectors
					cm = cm + abs(x)
				pn = 0
				for y in prod_vector:
					pn = pn + abs(y)

				div = cm*pn
				score = inner/div 		#final division
				score = score*rn 		#multiply with rn
				score = round(score,4)
				score_list.append(score)
				prod_list.append(p)

		sub_list = assign (prod_list, products, subclasses)	#get subclass of every product
		sub_list = [item for s in sub_list for item in s]
		sub_score = pd.DataFrame({'subclass':sub_list, 'score':score_list, 'product':prod_list}, columns=['subclass','score','product'])
		sub_score_score = sub_score.groupby('subclass')['score'].apply(list)
		sub_score_prod = sub_score.groupby('subclass')['product'].apply(list)
		selected_products = []
		selected_score = []
		count = 0
		for i in sub_score_score:
			ind = i.index(max(i))					#get max score for every subclass
			tmp = sub_score_prod[count]
			selected_products.append(tmp[ind])		#get product with max score
			selected_score.append(max(i))
			count = count+1

		cl_list = assign (prod_list, products, classes)		#get class of every product
		cl_list = [item for s in cl_list for item in s]
		cl_score = pd.DataFrame({'class':cl_list, 'score':score_list, 'product':prod_list}, columns=['class','score','product'])
		cl_score_score = cl_score.groupby('class')['score'].apply(list)
		cl_score_prod = cl_score.groupby('class')['product'].apply(list)
		count = 0
		for i in cl_score_score:
			ind = i.index(max(i))			#get first max score for every class
			tmp = cl_score_prod[count]
			selected_products.append(tmp[ind])
			selected_score.append(max(i))

			i.remove(max(i))
			tmp.remove(tmp[ind])
			if len(i)>0:
				ind = i.index(max(i))			#get second max score for every class (if exists)
				tmp = cl_score_prod[count]
				selected_products.append(tmp[ind])
				selected_score.append(max(i))
			count = count+1

		selected_cl = pd.DataFrame({'score':selected_score, 'product':selected_products}, columns=['score','product'])
		selected_cl = selected_cl.drop_duplicates()
		selected_cl = selected_cl.sort_values(by=['score'])		#list with selected items (1 for every subclass, 2 for every class)
		top = selected_cl['product'].tolist()
		rec_list.append(top[-10:])				#get top-10 products with highest score of selected items list
		count_id.append(out_count)		
		out_count = out_count + 1

	recommended = pd.DataFrame({'customer id':count_id,'top products':rec_list},columns=['customer id','top products'])
	return recommended
	
	

if __name__ == '__main__':

	#PREPROCESSING
	input_products = pd.read_csv("products-categorized.csv", header=None)	#read file products-categorized.csv
	pr = input_products[0]
	products = []
	classes = []
	subclasses = []
	j = 0

	for i in input_products[1]: 
		tokens = i.split("/")
		if len(tokens) > 1: 		#remove those products of unknown class and those with no subclass
			products.append(pr[j])
			classes.append(tokens[0])
			subclasses.append(tokens[1])
			j = j+1
		else:
			j = j+1

	categories = pd.DataFrame({'product':products, 'class':classes, 'subclass':subclasses}, columns=['product','class','subclass'])

	vec = CountVectorizer(token_pattern=r'\b[^\dW]+\b') 	#create vectors for subclasses
	subclass_vector = vec.fit_transform(subclasses).toarray()

	bask = []
	input_baskets = open('groceries.csv','r') 	#read basket file
	for line in input_baskets:
		bask.append(line.replace("\n",""))

	random_ids = [] 		#create random customer ids
	for i in range(0,len(bask)):
		r = random.randint(1,1000)
		random_ids.append(r)

	baskets = pd.DataFrame({'customer_id':random_ids, 'basket':bask}, columns=['customer_id','basket'])

	#CUSTOMER MODEL	
	agg_baskets = baskets.groupby('customer_id')['basket'].apply(list) 	#aggregate baskets of same customer_id
	agg_baskets = flat_baskets(agg_baskets)
	basket_out = pd.DataFrame({'agg_baskets':agg_baskets})
	basket_out.to_csv('baskets.csv')	
	cms = agg_vec(agg_baskets, products, subclass_vector)	#create customer vector and normalize it
	cms_norm = normalization(cms)

	#PRODUCT MODEL	 
	sub = assign(baskets['basket'],products,subclasses) 	#assign subclass/class to each product
	cl = assign(baskets['basket'],products,classes)
	#APRIORI - association rules in class/subclass level	
	rules_cl = association_rules(cl,0.05,0.3) 	#class assoctiaion rules	
	rules_sub = association_rules(sub,0.05,0.3) 	#subclass association rules	
	product_vectors = product_vector(subclasses,products,classes,rules_sub,rules_cl) 	#create customer vectors

	#SCORE	
	recommendations = score(agg_baskets, cms_norm, products, product_vectors, subclasses, classes) 	#score every product and keep top-10 for every customer
	recommendations.to_csv('recommendations.csv')
