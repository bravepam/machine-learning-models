#include"util.h"
#include"RandomTree.h"
#include<iostream>

using namespace std;

void testTreeDataSet()
{
	TreeDataSet tds(nullptr);
	tds.bagging(100);
	tds.oobData();
	for (size_t i = 0; i != tds.train_data.size(); ++i)
		cout << tds.train_data[i] << ' ';
	cout << endl << "oob size: " << tds.oob.size() << endl;
	for (size_t i = 0; i != tds.oob.size(); ++i)
		cout << tds.oob[i] << ' ';
	cout << endl;
}

int main()
{
	testTreeDataSet();
	getchar();
	return 0;
}