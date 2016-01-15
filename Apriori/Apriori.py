# -*- coding:utf-8 -*-

##############利用Apriori算法进行关联分析############

def loadData(): #产生数据集
	return [
	[1,3,4],
	[2,3,5],
	[1,2,3,5],
	[2,5]
	]

#以下筛选频繁集

def createCandidateSet1(dataset): #构造第一个候选集，
	cs1 = []
	for record in dataset:
		for item in record:
			if [item] not in cs1:
				cs1.append([item])
	cs1.sort()
	return list(map(frozenset,cs1))

def supportDegree(csk,dataset,minsupport): #计算支持度
	fscnt = {}
	for record in dataset: #对于每条记录
		for cs in csk: #对于每个候选集项
			if cs.issubset(record): #如果在该条记录出现
				fscnt[cs] = fscnt.get(cs,0) + 1
	m = float(len(dataset))
	fs = [];fssupport = {}
	for cs in fscnt: 
		support = fscnt[cs] / m #计算该项的支持度
		if support >= minsupport: #若达标
			fs.append(cs) #则记下
			fssupport[cs] = support
	return fs,fssupport

def createCandidateSetk(fsk_1,k): #根据频繁集k-1构造候选集k
	fsk = []
	m = len(fsk_1)
	for i in range(m):
		for j in range(i + 1,m):
			subset1 = list(fsk_1[i])[:k - 2];subset2 = list(fsk_1[j])[:k - 2] #筛选集合元素前缀，包含前k-2个元素
			subset1.sort();subset2.sort()
			if subset1 == subset2: #如果前缀相同
				fsk.append(fsk_1[i] | fsk_1[j]) #则合并，构造包含k个元素的候选集项
	fsk.sort()
	return fsk

def apriori(dataset,minsupport = 0.5): #Apriori算法
	cs1 = createCandidateSet1(dataset) #构造第一个候选集
	fs1, support = supportDegree(cs1,dataset,minsupport) #计算出第一个频繁集及支持度
	fs = [fs1]
	k = 2
	while(len(fs[k - 2]) > 0): #只要前一个频繁集非空
		csk = createCandidateSetk(fs[k - 2],k) #则根据其构造当前候选集
		fsk,supportk = supportDegree(csk,dataset,minsupport) #计算支持度等
		fs.append(fsk) #添加入频繁集列表
		support.update(supportk) #更新支持度字典
		k += 1
	return fs,support

#以下筛选关联规则

def caclConfidence(fs,Rk_1,support,rules,min_confidence): #计算关联规则可信度
	Rk = [] #下一个候选关联规则集合
	for item in Rk_1: #对于每个规则，这里Rk_1存储的是规则的右部，是集合列表
		conf = support[fs] / support[fs - item] #计算可信度，fs是频繁集项，也是集合
		if conf >= min_confidence:
			print(fs - item,"-->",item,"confidence:",conf)
			rules.append((fs - item,item,conf))
			Rk.append(item)
	return Rk

def generateRulesFromRk_1(fs,Rk_1,support,rules,min_confidence): #根据前一个关联规则集合产生当前候选规则集合
	m = len(Rk_1[0]) #前一个关联规则右部元素个数
	if(len(fs) > (m + 1)): #必须保证规则左边至少有一个元素
		Rk = createCandidateSetk(Rk_1,m + 1) #构造右部含有m+1个元素的规则
		Rkp1 = caclConfidence(fs,Rk,support,rules,min_confidence)
		if len(Rkp1) > 1: #若返回的关联规则集合不止一个规则
			generateRulesFromRk_1(fs,Rkp1,support,rules,min_confidence) #则继续递归构造右部元素更多的规则

def generateRules(frequentset,support,min_confidence = 0.7): #筛选满足条件的关联规则
	rules = []
	for i in range(1,len(frequentset)): #从1号频繁集开始，因为0号频繁集项只有一个元素
		for subset in frequentset[i]: #对于每个频繁集项
			Rk_1 = [frozenset([item]) for item in subset] #构造候选规则，只包含右部
			if i > 1: #若是2,3...号频繁集，则不必筛选右部只含一个元素的规则了，直接从含有两个元素的规则开始
				generateRulesFromRk_1(subset,Rk_1,support,rules,min_confidence)
			else: #否则
				caclConfidence(subset,Rk_1,support,rules,min_confidence)
	return rules

if __name__ == '__main__': #测试
	data = loadData()
	dataset = list(map(frozenset,data))
	fs,support = apriori(dataset,0.5)
	print(fs)
	#print(support)
	rules = generateRules(fs,support,0.7)
	print(rules)