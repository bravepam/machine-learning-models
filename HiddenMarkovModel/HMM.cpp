
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#include<vector>
#include<iostream>
#include<algorithm>

using namespace std;

class HMM
{//隐马尔科夫模型
private:
	vector<double> a; //初始状态概率
	vector<vector<double>> A; //状态转移概率矩阵
	vector<vector<double>> B; //状态生成观测概率矩阵
	const int N; //状态个数
	const int M; //观测值个数
private:
	vector<int> o; //此时计算所用的观测序列
private:
	vector<vector<double>> forward; //前向概率
	vector<vector<double>> backward; //后向概率
	vector<vector<double>> when_which; //某时刻出现某状态的概率
	vector<vector<vector<double>>> trans; //某时刻从某状态转移到另一状态的概率
private:
	template <typename T>
	void initMatrix(vector<vector<T>> &m, int r, int c)const
	{//给二维矩阵分配空间
		m.resize(r);
		for (int i = 0; i != r; ++i)
			m[i].resize(c);
	}
	template <typename T>
	void initMatrix(vector<vector<vector<T>>> &m, int d1, int d2, int d3)const
	{//给三维矩阵分配空间
		m.resize(d1);
		for (int i = 0; i != d1; ++i)
			initMatrix(m[i], d2, d3);
	}
	void HMM::initHMM()
	{//无监督学习模型前给模型参数设定初值
		const double trans_pro = 1.0 / N, ge_state_pro = 1.0 / M;
		fill_n(a.begin(), a.size(), trans_pro); //初值状态概率和为1
		for (int i = 0; i != N; ++i)
		{
			fill_n(A[i].begin(), A[i].size(), trans_pro); //从状态i转移到其他状态的概率和为1
			fill_n(B[i].begin(), B[i].size(), ge_state_pro); //从状态i生成观测值得概率和为1
		}
	}
	double vectorError(const vector<double> &v1, const vector<double> &v2)const
	{//两向量的误差平方和
		double sum = 0.0;
		for (int i = 0; i != v1.size(); ++i)
			sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		return sum;
	}

	//两次模型参数间的误差
	double error(const vector<vector<double>>&, const vector<vector<double>>&, 
		const vector<double>&)const;
public:
	HMM(int n, int m) :a(n), A(n, vector<double>(n)), B(n, vector<double>(m)), N(n), M(m){ initHMM(); }
	HMM(vector<double> &_a, vector<vector<double>> &_A, vector<vector<double>> &_B) :
		a(move(_a)), A(move(_A)), B(move(_B)), N(a.size()), M(B[0].size()){}
	//设置观测序列
	void setObservation(vector<int> &_o)
	{
		o = move(_o);
	}

	//probability computation
	//计算前向概率以得到观测序列出现的概率
	double forwardPro();
	//计算后向概率以得到观测序列出现的概率
	double backwardPro();
	//计算某时刻出现某状态的概率
	double whenAndWhichStatePro(int = -1, int = -1);
	//计算某时刻状态之间转移的概率
	double stateTransfer(int = -1, int = -1, int = -1);
	//出现某状态的期望
	double expectationOfstate(int);
	//从具体某状态转移到另一状态的期望
	double expectationTransFromState(int );
	//从状态1转移到状态2的期望
	double expectationTransItoJ(int, int);

	//learning
	//监督学习，即提供两棵观测值序列和状态序列
	void supervisedLearning(const vector<vector<int>>&, const vector<vector<int>>&);
	//无监督学习，只提供观测值序列
	void unsupervisedLearning(vector<int>&, double);

	//prediction
	//double optimalPathPro(vector<int>&);
	//维特比算法计算出现某一观测序列时的最佳状态序列，并返回概率
	double viterbi(vector<int>&, vector<int>&);
	//近似算法计算出现某一观测序列时的最佳状态序列，并返回概率
	double approximate(vector<int>&, vector<int>&);
};

double HMM::forwardPro()
{
	const int T = o.size();
	initMatrix(forward, T, N);
	for (int i = 0; i != N; ++i)
		forward[0][i] = a[i] * B[i][o[0]]; //初始0时刻时出现状态i的概率
	for (int t = 0; t != T - 1; ++t)
	{
		for (int i = 0; i != N; ++i)
		{
			for (int j = 0; j != N; ++j)
				//递推，时刻t+1出现i的概率，N个可能前驱都有可能转移到状态i
				forward[t + 1][i] += forward[t][j] * A[j][i];
			//乘以从状态i生成该时刻观测值的概率，即得到前向概率
			forward[t + 1][i] *= B[i][o[t + 1]]; 
		}
	}
	double sum = 0.0;
	for (int i = 0; i != N; ++i)
		sum += forward[T - 1][i];
	return sum; //观测序列出现的概率
}

double HMM::backwardPro()
{
	const int T = o.size();
	initMatrix(backward, T, N);
	for (int i = 0; i != N; ++i)
		backward[T - 1][i] = 1.; //初始，时刻末尾之后出现任何状态都行，即概率为1
	for (int t = T - 2; t >= 0; --t)
	{
		for (int i = 0; i != N; ++i)
		{
			for (int j = 0; j != N; ++j)
				//递推，
				backward[t][i] += A[i][j] * B[j][o[t + 1]] * backward[t + 1][j];
		}
	}
	double sum = 0.0;
	for (int i = 0; i != N; ++i)
		sum += a[i] * B[i][o[0]] * backward[0][i];
	return sum; //观测序列出现的概率
}

double HMM::whenAndWhichStatePro(int time, int state)
{
	if (!when_which.empty())
		return when_which[time][state];
	if (forward.empty() || backward.empty())
	{//就算前向和后巷概率
		forwardPro();
		backwardPro();
	}
	const int T = o.size();
	initMatrix(when_which, T, N);
	double sum = 0.0;
	for (int j = 0; j != N; ++j)
		sum += forward[0][j] * backward[0][j]; //在时刻t出现任一状态的概率之和相等

	for (int t = 0; t != T; ++t)
		for (int i = 0; i != N; ++i)//时刻t出现状态i的概率除以时刻t出现状态的总概率即为...
			when_which[t][i] = forward[t][i] * backward[t][i] / sum;
	if (time < 0) return 0.0;
	return when_which[time][state];
}

double HMM::stateTransfer(int time, int state1, int state2)
{
	if (!trans.empty()) return trans[time][state1][state2];
	if (forward.empty() || backward.empty())
	{
		forwardPro();
		backwardPro();
	}
	const int T = o.size();
	initMatrix(trans, T, N, N);
	for (int t = 0; t != T; ++t)
	{
		double sum = 0.0;
		for (int i = 0; i != N; ++i)
		{
			for (int j = 0; j != N; ++j)
			{
				//时刻t从状态i转移到状态j的概率
				trans[t][i][j] = forward[t][i] * A[i][j] * B[j][o[t]] * backward[t][j];
				sum += trans[t][i][j]; //时刻t发生状态转移的总概率概率
			}
		}
		for (int i = 0; i != N; ++i)
		{
			for (int j = 0; j != N; ++j)
				trans[t][i][j] /= sum; //相比即为时刻t从状态i转移到状态j的概率，条件概率
		}
	}
	if (time < 0) return 0.0;
	return trans[time][state1][state2];
}

double HMM::expectationOfstate(int state)
{
	double sum = 0.0;
	const int T = o.size();
	for (int t = 0; t != T; ++t)
		sum += when_which[t][state];
	return sum;
}

double HMM::expectationTransFromState(int state)
{
	double sum = 0.0;
	const int T = o.size();
	for (int t = 0; t != T - 1; ++t)
		sum += when_which[t][state];
	return sum;
}

double HMM::expectationTransItoJ(int state_I, int state_J)
{
	double sum = 0.0;
	const int T = o.size();
	for (int t = 0; t != T - 1; ++t)
		sum += trans[t][state_I][state_J];
	return sum;
}

void HMM::supervisedLearning(const vector<vector<int>> &obs, const vector<vector<int>> &state)
{//监督学习
	const int S = obs.size(), T = obs[0].size();
	vector<double> sum_trans(N), sum_obs(N);
	for (int s = 0; s != S; ++s)
	{
		for (int t = 0; t != T - 1; ++t)
		{
			++A[state[s][t]][state[s][t + 1]];
			++B[state[s][t]][obs[s][t]];
			++sum_trans[state[s][t]];
			++sum_obs[state[s][t]];
		}
		++a[state[s][0]];
	}
	for (int i = 0; i != N; ++i)
	{
		for (int j = 0; j != N; ++j)
			A[i][j] /= sum_trans[i]; //从状态i转移到状态j的频数除以从状态i发生转移的频数即为转移概率
	}
	for (int i = 0; i != N; ++i)
	{
		for (int k = 0; k != M; ++k)
			B[i][k] /= sum_obs[i]; //从状态i生成观测值k的频数比上从状态i生成观测值得频数即为生成概率
	}
	for (int i = 0; i != N; ++i)
		a[i] /= S; //时刻0状态i出现的频数比上总样本数即为初始状态概率
}

void HMM::unsupervisedLearning(vector<int> &_o, double eps)
{//无监督学习
	const int T = _o.size();
	setObservation(_o);
	vector<double> sum1(N), tempa(N);
	vector<vector<double>> sum2, sum3, tempA, tempB;
	initMatrix(sum2, N, N); initMatrix(sum3, N, M);
	initMatrix(tempA, N, N); initMatrix(tempB, N, M);
	while (true)
	{
		whenAndWhichStatePro(); //根据当前模型参数计算某时刻出此案某状态的概率
		stateTransfer(); //计算状态转移概率
		//根据EM算法计算新的模型参数
		for (int t = 0; t != T; ++t)
		{
			for (int i = 0; i != N; ++i)
			{
				sum1[i] += when_which[t][i];
				for (int j = 0; j != N; ++j)
					sum2[i][j] += trans[t][i][j];
				for (int k = 0; k != M; ++k)
					sum3[i][k] += when_which[t][i] * static_cast<double>(k == o[t]);
			}
		}
		for (int i = 0; i != N; ++i)
		{
			for (int j = 0; j != N; ++j)
				tempA[i][j] = sum2[i][j] / sum1[i];
			for (int k = 0; k != M; ++k)
				tempB[i][k] = sum3[i][k] / sum1[i];
			tempa[i] = when_which[0][i];
			fill_n(sum2[i].begin(), sum2[i].size(), 0.0);
			fill_n(sum3[i].begin(), sum3[i].size(), 0.0);
		}
		fill_n(sum1.begin(), sum1.size(), 0.0);
		double e = error(tempA, tempB, tempa); //比较误差
		A.swap(tempA), B.swap(tempB), a.swap(tempa);
		if (e < eps) break; //若已到期望精度，则退出
	}
	cout << "learning done..." << endl;
}

double HMM::error(const vector<vector<double>> &tempA, const vector<vector<double>> &tempB,
	const vector<double> &tempa)const
{
	double sum = 0.0;
	for (int i = 0; i != tempA.size(); ++i)
		sum += vectorError(tempA[i], A[i]);
	for (int j = 0; j != tempB.size(); ++j)
		sum += vectorError(tempB[j], B[j]);
	sum += vectorError(tempa, a);
	return sqrt(sum);
}

double HMM::approximate(vector<int> &_o, vector<int> &res)
{//近似算法求产生观测序列o的最优状态序列
	setObservation(_o);
	whenAndWhichStatePro(); //需要用到某时刻出现某状态的概率矩阵
	const int T = o.size();
	double max = 0.0, pro = 1.0;
	int state = -1;
	for (int t = 0; t != T; ++t)
	{
		for (int i = 0; i != when_which[t].size();++i)
			if (when_which[t][i] > max)
			{//记下某时刻出现最可能的状态
				max = when_which[t][i];
				state = i;
			}
		pro *= max;
		res.push_back(state); //即为所求状态
	}
	return pro; //返回该状态出现的概率
}

double HMM::viterbi(vector<int> &_o, vector<int> &res)
{//维特比算法计算产生观测序列o的最优状态序列
	setObservation(_o);
	const int T = o.size();
	//记录在时刻t状态为i的所有路径中概率的最大值,max_pro[t][i]
	vector<vector<double>> max_pro;
	//记录在时刻t状态为i的最大概率路径中的第t - 1个节点，即能够最大概率转移到状态i的某一状态,node[t][i]
	vector<vector<int>> node;
	initMatrix(max_pro, T, N), initMatrix(node, T, N);
	for (int i = 0; i != N; ++i)
	{
		//初始时刻取得状态i的概率
		max_pro[0][i] = a[i] * B[i][o[0]]; 
		node[0][i] = -1; //无前驱
	}
	for (int t = 1; t != T; ++t)
	{//在时刻t
		for (int i = 0; i != N; ++i)
		{//状态为i的最大概率路径
			double max = 0.0;
			int node_id = -1;
			for (int j = 0; j != N;++j) //对每一个可能前驱
				if (max_pro[t - 1][j] * A[j][i] > max)
				{//计算概率，记下概率最大的
					max = max_pro[t - 1][j] * A[j][i];
					node_id = j;
				}
			max_pro[t][i] = max * B[i][o[t]]; //存入矩阵
			node[t][i] = node_id; //并得到前驱
		}
	}
	double max = 0.0;
	int last_node = -1;
	for (int i = 0; i != N;++i)
		if (max_pro[T - 1][i] > max)
		{//对最终时刻扫描每一个可能状态，记下概率最大的
			max = max_pro[T - 1][i];
			last_node = i;
		}
	res.push_back(last_node); //即为状态序列最后一个状态
	for (int t = T - 1; t > 0; --t)
	{//从最后时刻开始，利用node矩阵得到每一个前驱
		res.push_back(node[t][last_node]);
		last_node = node[t][last_node];
	}
	reverse(res.begin(), res.end()); //逆置，即得到状态序列
	return max;
}

int main()
{
	//vector<vector<double>> A = { { 0.5, 0.2, 0.3 }, { 0.3, 0.5, 0.2 }, { 0.2, 0.3, 0.5 } };
	//vector<vector<double>> B = { { 0.5, 0.5 }, { 0.4, 0.6 }, { 0.7, 0.3 } };
	//vector<double> a = { 0.2, 0.4, 0.4 };
	//HMM hmm(a, A, B);
	//vector<int> o = { 0, 1, 0, 0, 1, 0, 1, 1 };
	//hmm.setObservation(o);
	//cout << hmm.forwardPro() << endl;
	//cout << hmm.backwardPro() << endl;
	//cout << hmm.whenAndWhichStatePro(4, 2) << endl;
	//cout << hmm.stateTransfer(2,1,2) << endl;

	//vector<int> o = { 0, 1, 0 }, res;
	//cout << hmm.approximate(o, res) << endl;
	//for (int i = 0; i != res.size(); ++i)
	//cout << res[i] << ' ';
	//cout << endl;
	//cout << hmm.viterbi(o, res) << endl;
	//for (int i = 0; i != res.size(); ++i)
	//cout << res[i] << ' ';
	//cout << endl;

	HMM hmm(3, 2);
	vector<int> o = { 0, 1, 0, 0, 1, 0, 1, 1 };
	hmm.unsupervisedLearning(o, 0.1);
	getchar();
	return 0;
}
