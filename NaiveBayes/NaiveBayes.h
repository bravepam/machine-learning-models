#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

#include<vector>

using namespace std;

class NaiveBayes
{/*朴素贝叶斯模型
 *k记为类别的第k个取值
 *j表示第j个特征
 *v表示某个特征的第v个取值，其中k,j,v = 0,1....K-1,J-1,Sj-1
 *Sj表示第j个特征的取值个数
 */
private:
	/*probability存储在第k个类别值的情况下，第j个特征取第v个值的概率，
	*即，P(Xj = xjv | Y = k)，因而它的大小为K*(S0+S1+...+Sj+...),j = 0,1...J-1
	*/
	vector<vector<vector<double>>> probability;
	vector<double> cls;//cls[k]表示取第k个类别值的概率
	double lambda;//贝叶斯估计采用的参数，避免某些项概率为0
public:
	NaiveBayes() :lambda(0.0), cls(), probability(){}
	void create(vector<int>&, int, double);//创建模型
	void train(vector<vector<int>>&, vector<int>&);//训练模型
	int compute(vector<int>&, double&);//计算某样本点所属的类别
};

void NaiveBayes::create(vector<int> &fvn, int clsn, double _lambda)
{
	/*创建模型，主要是分配空间
	*fvn，即feature-value-number，fvn[j]表示第j个特征有几个取值
	*clsn，即class-number，表示有几个类别值
	*/
	lambda = _lambda;
	cls.resize(clsn);
	probability.resize(clsn);
	int feature_number = fvn.size();
	for (int k = 0; k != probability.size(); ++k)
	{
		probability[k].resize(feature_number);
		for (int j = 0; j != probability[k].size(); ++j)
			probability[k][j].resize(fvn[j]);
	}
}

void NaiveBayes::train(vector<vector<int>> &smpps, vector<int> &smpvs)
{
	/*训练模型
	*smpps,即sample points，表示样本点
	*smpvs,即sample values，表示对应地样本值
	*/
	//累计样本值
	int sample_number = smpvs.size();
	for (int si = 0; si != sample_number; ++si)
	{
		++cls[smpvs[si]];//样本值为smpvs[si]的样本数加1
		//在样本（类别）值为smpvs[si]的前提下，第j个特征取值为smpps[si][j]的数量加1
		for (int j = 0; j != probability[smpvs[si]].size(); ++j)
			++probability[smpvs[si]][j][smpps[si][j]];
	}
	//计算概率
	for (int k = 0; k != cls.size(); ++k)
	{
		for (int j = 0; j != probability[k].size(); ++j)
			for (int v = 0; v != probability[k][j].size(); ++v)
				probability[k][j][v] = //计算P(Xj = xjv | Y = k)
				(probability[k][j][v] + lambda) / (cls[k] + probability[k][j].size() * lambda);
		cls[k] = (cls[k] + lambda) / (sample_number + cls.size() * lambda);//计算P(Y = k)
	}
}

int NaiveBayes::compute(vector<int> &data, double &pro)
{//计算数据点data所属的类别和相对应的概率
	int data_cls = -1;
	pro = 0.0;
	for (int k = 0; k != cls.size(); ++k)
	{
		//计算在取第k个类别值的情况下，data中各个特征分量在相对应的取值的概率的
		//乘积，再乘以类别为kth后所得的结果，即为所求概率，公式由贝叶斯公式推得
		//即P(Y=k)P(Xj=xj | Y=k),j = 0,1...Sj-1
		double temp_pro = cls[k];
		for (int j = 0; j != data.size(); ++j)
			temp_pro *= probability[k][j][data[j]];
		if (temp_pro > pro)
		{//取其中最大者
			data_cls = k;
			pro = temp_pro;
		}
	}
	return data_cls;
}

#endif