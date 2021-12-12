#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <cassert>
#include <iostream>
#include<string>
#include<fstream>
#include <limits>
#include <queue>


using namespace std;
using namespace Eigen;

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




Eigen::MatrixXd load_csv(std::string path_file, int rows, int cols) {

	std::ifstream reader(path_file);

	std::string line_stream;

	int row = 0;
	int col = 0;

	Eigen::MatrixXd result = Eigen::MatrixXd(rows, cols);
	if (reader.is_open()) {

		while (std::getline(reader, line_stream)) {

			char* ptr = (char*)line_stream.c_str();
			int len = line_stream.length();
			col = 0;

			char* start = ptr;
			for (int i = 0; i < len; i++) {

				if (ptr[i] == ',') {
					result(row, col++) = atof(start);
					start = ptr + i + 1;
				}
			}
			result(row, col) = atof(start);

			row++;
		}

		reader.close();
	}
	return result;
}
MatrixXd Sequence_generator(float font, float end, int step)
{
	MatrixXd result(1, step);
	float step_size = (log(end) - log(font)) / (step - 1);
	//result.push_back(font);
	for (int i = 0; i < step; i++)
		result(0, i) = (exp(log(font) + i * step_size));
	return result;
}

Eigen::MatrixXd hks_computing(string name)

{
	int time_sample = 100;


	cout << "start to calculate the HKS of " + name + ".obj" << endl;

	MatrixXd eg_val = load_csv(root_path + "Results/" + name + "_eigval.csv", 300, 1);
	MatrixXd eg_vect = load_csv(root_path + "Results/" + name + "_eigvec.csv", 6890, 300);
	MatrixXd diag = load_csv(root_path + "Results/" + name + "_diag.csv", 6890, 6890);
	

	float tmin = abs(4 * log(10) / eg_val(eg_val.rows()-1, 0));
	float tmax = abs(4 * log(10) / eg_val(0, 0));

	float start = tmin;
	MatrixXd result = MatrixXd::Zero(6890,time_sample);
	MatrixXd seq = Sequence_generator(tmin, tmax, time_sample);// generate the logmatic time sequence

	eg_vect = eg_vect.array().abs().square();//6890*300
	//eg_val=-abs(eg_val.array());
	eg_val = abs(eg_val(0,0))-abs(eg_val.array());//300*1

	result = eg_vect * ((eg_val*seq).array().exp().matrix());

	MatrixXd colsum = (diag*result).colwise().sum();//6890*6890\times 6890*100==>1*100
	Eigen::SparseMatrix<double> scale(time_sample,time_sample);
	for (int i = 0; i < time_sample; i++)
		scale.insert(i, i) = 1/colsum(0, i);

	result = result * scale;
	saveData(root_path + "Results/" + name + "_hks.csv", result);
	return result;
};

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

tuple<vector<int>, float> Point_correspondence(const Eigen::MatrixXd &source, const Eigen::MatrixXd &target)
{
	float error = 0.;
	vector<int> keypoint = read_keypoint();
	vector<int> res;
	for (int i = 0; i < keypoint.size(); i++)
	{
		int s_dice = keypoint[i];
		MatrixXd s_point = source.row(s_dice);
		float tmp_error = std::numeric_limits<float>::infinity();
		int indice = 0;
		for (int j = 0; j < target.rows(); j++)
		{

			float tmp = (s_point - target.row(j)).squaredNorm();//calculate the L_{2} distance between the HKS vectors 
			//cout << tmp << endl;
			if (tmp < tmp_error)
			{
				cout << j << endl;
				tmp_error = tmp;
				indice = j;
			}
		}
		//cout << tmp_error << endl;
		error += tmp_error;
		res.push_back(indice);
	}
	return make_tuple(res, error);
}


void Question3(string source_name, string target_name)
{
	Eigen::MatrixXd hks_source = hks_computing(source_name);//get the HKS of the source mesh
	Eigen::MatrixXd hks_target = hks_computing(target_name);// get the HKS of the target mesh
	//cout << hks_source.row(0) << endl;
	vector<int> cor_map;
	float error;
	cout << "start to matching " + source_name + "\n" << endl;
	tie(cor_map, error) = Point_correspondence(hks_source, hks_target);
	ofstream file;
	file.open(root_path + "Results/" + source_name + "_hks_corr_vids.txt");
	for (auto x : cor_map)
		file << to_string(x) + "\n";
	file.close();

}
void Report_total_error()
{
	cout << "start to evaluate all source meshes" << endl;
	Eigen::MatrixXd target = hks_computing("target");
	string source_name;

	for (int i = 1; i < 10; i++)
	{
		ofstream file;
		source_name = "source_" + to_string(i);
		Eigen::MatrixXd source = hks_computing(source_name);
		vector<int> cor_map;
		float error;
		cout << "start to matching " + source_name + "\n" << endl;
		tie(cor_map, error) = Point_correspondence(source, target);

		file.open(root_path + "Results/" + source_name + "_hks_corr_vids.txt");
		for (auto x : cor_map)
			file << to_string(x) + "\n";
		file.close();
	}

}
int main(int argc, char** argv) {
	string source_name= argv[1];
	string target_name = argv[2];
	//float error=Question3(source_name,target_name);
	Report_total_error();


	return 0;
}
