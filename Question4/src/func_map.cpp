#include <Eigen/Core>
#include <Eigen/Sparse>
#include<Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <cassert>
#include <iostream>
#include<string>
#include <Eigen/SVD>
#include<fstream>
#include <limits>
#include <queue>
#include<Tools.h>


using namespace std;
using namespace Eigen;


// using the definition of HW3
string root_path = "/root/original/HW5/";

void saveData(string fileName, MatrixXd  matrix)
{
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

vector<int> read_keypoint()
{
	vector<int> keypoints;

	std::ifstream file(root_path + "data/keypoint_vids.txt");

	// to save the key_point information and its adjacent edges.

	int number;

	if (true == file.is_open()) {
		std::string s;
		while (file >> number) {
			keypoints.push_back(number);
		}

		file.close();
	}
	else {
		std::cout << "file open fail" << endl;
	}
	return keypoints;
}


tuple<MatrixXd, float> solver(MatrixXd p1, MatrixXd p2, VectorXd L1, VectorXd L2, float lambda)

{
	//implement the fitting of ||C*p1-p2||+lambda*||C*L1-L2*C||
	int n1 = p1.rows();
	int n2 = p2.rows();
	MatrixXd A_fixed = p1 * p1.transpose();
	MatrixXd B = p1 * p2.transpose();

	MatrixXd X = MatrixXd::Zero(n2, n1);

	for (int i = 0; i < n2; i++)
	{
		VectorXd test = lambda * (L1.array() - L2(i)).square();
		Eigen::MatrixXd diag = test.array().matrix().asDiagonal();
		diag += A_fixed;

		VectorXd sol(Map<VectorXd>(B.col(i).data(), B.rows()));
		X.row(i) = diag.bdcSvd(ComputeThinU | ComputeThinV).solve(sol);


	}
	float residual = ((X*p1) - p2).norm();
	residual *= residual;
	return make_tuple(X, residual);
}

vector<int> Matching(MatrixXd FM, MatrixXd s_ps, MatrixXd t_ps)
{
	t_ps = t_ps.transpose();
	int nbr_s = s_ps.rows();
	int nbr_t = t_ps.rows();
	vector<int> result;
	vector<int> keypoint = read_keypoint();

	MatrixXd s_to_t = (FM * s_ps).transpose();//40*6890

	assert(s_to_t.size() == 40 * 6890);
	//cout<<keypoint.size()<<endl;
	for (int i = 0; i < keypoint.size(); i++)
	{
		int s_dice = keypoint[i];
		MatrixXd s_point = s_to_t.row(s_dice);
		float tmp_error = std::numeric_limits<float>::infinity();
		int indice = 0;

		for (int j = 0; j < nbr_t; j++)
		{

			float tmp = (s_point - t_ps.row(j)).squaredNorm();//calculate the L_{2} distance between the HKS vectors 
			//cout<<tmp<<endl;
			if (tmp < tmp_error)
			{
				//cout<<tmp<<endl;
				tmp_error = tmp;
				indice = j;
			}
		}
		result.push_back(indice);
	}
	cout << result.size() << endl;
	return result;

}

void FM_matching(string source, string target, int K)
{

	// First get the function representation of the source (A) and target (B)

	MatrixXd F = load_csv(root_path + "Results/" + source + "_hks.csv", 6890, 100);//dim of the descriptor K=100
	MatrixXd G = load_csv(root_path + "Results/" + target + "_hks.csv", 6890, 100);


	MatrixXd S = load_csv(root_path + "Results/" + source + "_eigvec.csv", 6890, 300);
	MatrixXd T = load_csv(root_path + "Results/" + target + "_eigvec.csv", 6890, 300);


	MatrixXd S_eigval = load_csv(root_path + "Results/" + source + "_eigval.csv", 300, 1);
	MatrixXd T_eigval = load_csv(root_path + "Results/" + target + "_eigval.csv", 300, 1);

	Map<MatrixXd, 0, OuterStride<> > s_base(S.data(), S.rows(), K, OuterStride<>(S.outerStride()));//6890*40
	Map<MatrixXd, 0, OuterStride<> > t_base(T.data(), T.rows(), K, OuterStride<>(T.outerStride()));//6890*40

	// Get the pseudo inverse of the bases
	MatrixXd trans_m_bases = s_base.completeOrthogonalDecomposition().pseudoInverse();//40*6890
	MatrixXd trans_n_bases = t_base.completeOrthogonalDecomposition().pseudoInverse();//40*6890

	string source_path = root_path + "data/" + source + ".obj";
	string target_path = root_path + "data/" + target + ".obj";


	Map<VectorXd, 0, OuterStride<> > s_diag(S_eigval.data(), K, S_eigval.cols(), OuterStride<>(S.outerStride()));//6890*40
	Map<VectorXd, 0, OuterStride<> > t_diag(T_eigval.data(), K, T_eigval.rows(), OuterStride<>(T.outerStride()));//6890*40

	MatrixXd A = trans_m_bases * F;//40*100
	MatrixXd B = trans_n_bases * G;//40*100

	//solve the Least square function
	MatrixXd FM;
	float error;
	tie(FM, error) = solver(A, B, s_diag, t_diag, 5);//Calculate the functional map and corresponding error of the fitting.

	assert(FM.size() == K * K);

	saveData(root_path + "Results/FM_" + source + "_func_map.csv", FM);

	vector<int> res = Matching(FM, trans_m_bases, trans_n_bases);
	ofstream file;
	file.open(root_path + "Results/" + source + "_fm_corr_vids.txt");
	for (auto x : res)
		file << to_string(x) + "\n";
	file.close();


}

void Report_total_error()
{
	cout << "start to evaluate all source meshes" << endl;
	//Eigen::MatrixXd target = hks_computing("target");
	string source_name;

	for (int i = 1; i < 10; i++)
	{
		ofstream file;
		source_name = "source_" + to_string(i);
		FM_matching(source_name, "target", 40);
	}

}


int main(int i, char**argv) {
	string source_name = argv[1];
	string target_name = argv[2];
	//FM_matching(source_name,target_name);	
	Report_total_error();

	return 0;
}
