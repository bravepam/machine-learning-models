#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include"RandomTree.h"
#include"util.h"
#include<memory>

//随机森林
class RandomForest
{
private:
	std::vector<std::shared_ptr<RandomTree>> rf; //随机树集合
	std::shared_ptr<const RFParams> prf = nullptr; //随机森林参数
	std::vector<const TreeDataSet*> datasets; //各棵树所用的bagging数据集
	std::vector<std::pair<size_t, double>> fis; //特征重要性，{特征，重要性度量值}
	double gen_err = 0.0; //泛化误差
	double test_err = 0.0; //测试误差
	double oob_err = 0.0; //随机树的泛化误差
public:
	RandomForest() :rf(), datasets(){}
	void setParams(std::shared_ptr<RFParams>&);
	void train();
	int predict(const sample&)const;
	double testError();
	double generalizationError();
	//随机树的带外误差均值
	double avgOobErrorOfTree();
	const std::vector<std::pair<size_t, double>>& FeatureImportance();
	static std::vector<sample> loadData(const std::string&, size_t, size_t&);
};

#endif