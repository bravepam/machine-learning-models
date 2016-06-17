#include"RandomForest.h"
#include"util.h"
#include<hash_map>

void RandomForest::train()
{
	for (size_t i = 0; i != prf->N; ++i)
	{
		std::shared_ptr<RandomTree> prt = std::make_shared<RandomTree>(prf);
		const TreeDataSet* ptds = prt->create();
		rf.emplace_back(prt);
		datasets.push_back(ptds);
	}
}

int RandomForest::predict(const sample& s)const
{
	std::hash_map<int, size_t> cls_count;
	for (size_t i = 0; i != rf.size(); ++i)
	{
		++cls_count[rf[i]->predict(s)];
	}

	std::pair<int, size_t> max{ INT_MIN, 0 };
	for (auto iter = cls_count.begin(); iter != cls_count.end(); ++iter)
	{
		if (iter->second > max.second)
			max = *iter;
	}
	return max.first;
}

double RandomForest::testError()
{
	size_t error = 0;
	for (size_t i = 0; i != prf->test_set.size(); ++i)
	{
		const int pred = predict(prf->test_set[i]);
		error += static_cast<size_t>(pred != prf->test_set[i].y);
	}
	test_err = error * 1.0 / prf->test_set.size();
	return test_err;
}

double RandomForest::genError()
{
	std::vector<double> oob_errors;
	double sum_error = 0.0;
	const std::vector<sample>& train = prf->train_set;
	oob_errors.reserve(train.size());
	for (size_t i = 0; i != train.size(); ++i)
	{
		size_t error = 0, howmany_trees = 0;
		for (size_t j = 0; j != rf.size(); ++j)
		{
			if (!datasets[j]->contains(i))
			{
				const int pred = rf[j]->predict(train[i]);
				error += static_cast<size_t>(pred != train[i].y);
				++howmany_trees;
			}
		}
		const double temp = error * 1.0 / howmany_trees;
		sum_error += temp;
		oob_errors.push_back(temp);
	}
	gen_err = sum_error / oob_errors.size();
	return gen_err;
}