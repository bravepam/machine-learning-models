#include<vector>
#include<fstream>
#include"Cart.h"

using namespace std;
const int dim = 4;

int main()
{
	ifstream infile("ddata.txt");
	if (!infile)
	{
		cout << "open file error" << endl;
		getchar();
		return -1;
	}
	int v, cnt = 0;
	vector<int> temp;
	vector<sample<int>> data;
	while (infile >> v)
	{
		if (v == -1) break;
		temp.push_back(v);
		++cnt;
		if (cnt == dim)
		{
			infile >> v;
			data.push_back(sample<int>(temp, v));
			cnt = 0;
		}
	}
	vector<int> feature_values = { 3, 2, 2, 3 };
	Cart cart(data, feature_values, 2, 0.1, 2);
	cart.create();
	vector<int> vec = { 1, 1, 0, 0 };
	cout << cart.compute(vec) << endl;
	getchar();
	return 0;
}