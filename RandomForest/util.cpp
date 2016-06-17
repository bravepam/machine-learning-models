#include"util.h"
#include<ctime>
#include<algorithm>

void randomVectorEngine(size_t num, size_t max, std::vector<size_t>& vec)
{
	//采用线性同余获得随机数，X(i+1) = {X(i) * A + C} mod B,此处A = 16807,C = 0,B = 2147483647(2^31 - 1)
	using lce = std::linear_congruential_engine < unsigned long, 16807, 0, 2147483647 >;
	time_t t;
	time(&t);
	srand(static_cast<unsigned int>(t));
	unsigned int seed = 0;
	while (seed == 0)
		seed = rand();
	lce rnd(seed);
	vec.reserve(num);
	for (size_t i = 0; i != max; ++i)
	{
		vec.push_back(rnd() % max);
	}
}

void TreeDataSet::bagging(size_t data_size)
{
	randomVectorEngine(data_size, data_size, train_data);
}

const std::vector<size_t>& TreeDataSet::getOobData()
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