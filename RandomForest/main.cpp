#include"RandomForest.h"
#include<iostream>

#define PREFIX R"(E:\Github\machine-learning-models\RandomForest\data\)"

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

void testRandomForestWithEcoli()
{
	RandomForest rf;
	Termcriteria tc(0.8, 5, 0, 0);
	size_t train_size = 336;
	std::shared_ptr<RFParams> rfp = newRFParams(
		RandomForest::loadData(PREFIX"ecoli_train.txt", 7, train_size), //训练集
		//RandomForest::loadData("test.txt", 8, test_size), //测试集
		vector<sample>(), //空表示无测试集
		8, //类别数
		7, //数据维度
		2, //训练随机树时每个节点可选的特征数
		10, //树的数目
		tc, //终止条件
		true //是否计算特征重要性
		);
	rf.setParams(rfp);
	rf.train();
	cout << "size: " << train_size << endl;
	const double avg_oob_error = rf.avgOobErrorOfTree();
	cout << "average OOB error of all Random Trees: " << avg_oob_error << endl;
	const double gen_error = rf.generalizationError();
	cout << "generalization error: " << gen_error << endl;
	auto fi = rf.featureImportance();
	for (size_t i = 0; i != fi.size(); ++i)
	{
		cout << "Feature ID: " << fi[i].first << " Importance: " << fi[i].second << endl;
	}
}

void testRandomForestWithSonar()
{
	RandomForest rf;
	Termcriteria tc(0.1, 4, 0, 0);
	size_t train_size = 208;
	std::shared_ptr<RFParams> rfp = newRFParams(
		RandomForest::loadData(PREFIX"sonar_train.txt", 60, train_size),
		//RandomForest::loadData("test.txt", 8, test_size),
		vector<sample>(),
		2,
		60,
		8,
		100,
		tc,
		true
		);
	rf.setParams(rfp);
	rf.train();
	cout << "size: " << train_size << endl;
	const double avg_oob_error = rf.avgOobErrorOfTree();
	cout << "average OOB error of all Random Trees: " << avg_oob_error << endl;
	const double gen_error = rf.generalizationError();
	cout << "generalization error: " << gen_error << endl;
	auto fi = rf.featureImportance();
	for (size_t i = 0; i != fi.size(); ++i)
	{
		cout << "Feature ID: " << fi[i].first << " Importance: " << fi[i].second << endl;
	}
}

int main()
{
	//testTreeDataSet();
	testRandomForestWithEcoli();
	//testRandomForestWithSonar();
	getchar();
	return 0;
}