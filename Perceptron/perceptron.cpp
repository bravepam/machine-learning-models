
/*******************************************
* Author: bravepam
*
* E-mail:1372120340@qq.com
*******************************************
*/


#include<sstream>
#include<iostream>
#include<vector>
#include"perceptron.h"

using namespace std;

int main()
{
	perceptron p(1.0, 2);
	p.initData("data.txt");
	vector<float> w(2, 0);
	float b = 0.0;
	p.compute(w, b);
	stringstream ss;
	ss << "y = sign(";
	for (int i = 0; i != 2; ++i)
		ss << w[i] << " * x" << i << " + ";
	ss << b << ')';
	cout << ss.str() << endl;
	getchar();
	return 0;
}