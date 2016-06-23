#include"util.h"
#include<Windows.h>
#include<algorithm>
#include<iostream>

unsigned int prev_seed = 0;

std::mt19937 getMt19937()
{
	srand(GetTickCount()); //用系统毫秒时间做种子
	unsigned int seed = 0;
	while (seed == prev_seed || seed == 0)
	{
		seed = rand(); //获得一个随机非0种子
		//printf("in loop, seed = %u\n", seed);
	}
	prev_seed = seed;
	//printf("seed = %u---------------\n", seed);
	std::mt19937 mt(seed);
	return mt;
}

void samplingWithReplacement(size_t num, size_t max, std::vector<size_t>& vec)
{
	std::mt19937 mt = getMt19937();
	vec.reserve(num);
	for (size_t i = 0; i != max; ++i)
	{
		vec.push_back(mt() % max);
	}
}

void samplingNoReplacement(size_t num, size_t max, std::vector<size_t>& vec)
{
	std::mt19937 mt = getMt19937();
	vec.reserve(num);
	std::vector<bool> selected(max, false);
	size_t count = 0;
	while (count < num)
	{
		size_t temp = mt() % max;
		if (!selected[temp]) //如果未选取了该数据
		{
			vec.push_back(temp);
			selected[temp] = true;
			++count;
		}
	}
}

std::shared_ptr<RFParams> newRFParams(const std::vector<sample>& train,
	const std::vector<sample>& test,
	size_t cn, size_t d, size_t f, size_t n,
	const Termcriteria& tc,
	bool cvi)
{
	return std::make_shared<RFParams>(
		std::move(train),
		std::move(test),
		d, cn, f, n, cvi,
		tc
		);
}

void TreeDataSet::bagging(size_t data_size)
{
	samplingWithReplacement(data_size, data_size, train_data);
}

const std::vector<size_t>& TreeDataSet::oobData()
{
	const size_t data_size = train_data.size();
	sort(train_data.begin(), train_data.end());
	oob.reserve(data_size);

	size_t i = 0, index = 0;
	while (i != data_size && index != data_size)
	{
		if (i < train_data[index])
			oob.push_back(i);
		else if (i == train_data[index])
		{
			++index;
			while (index != data_size && i == train_data[index])
				++index;
		}
		++i;
	}
	for (; i != data_size; ++i)
		oob.push_back(i);
	oob.resize(oob.size());
	return oob;
}

bool TreeDataSet::contains(size_t sample_id)const
{
	auto iter = std::find(oob.begin(), oob.end(), sample_id);
	return (iter == oob.end());
}