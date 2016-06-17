#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include"RandomTree.h"
#include"util.h"
#include<memory>

class RandomForest
{
private:
	std::vector<std::shared_ptr<RandomTree>> rf;
	std::shared_ptr<const RFParams> prf = nullptr;
	std::vector<const TreeDataSet*> datasets;
	double gen_err = -INT_MAX;
	double test_err = -INT_MAX;
public:
	RandomForest() :rf(), datasets(){}
	void setParams(const RFParams* p)
	{
		prf.reset(p);
		rf.reserve(p->N);
		datasets.reserve(p->N);
	}
	void train();
	int predict(const sample&)const;
	double testError();
	double genError();
};

#endif