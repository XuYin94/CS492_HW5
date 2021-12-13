#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <Eigen/Core>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Utils/getopt.h>
#include <iostream>
#include<string>
#include<fstream>
#include <limits>
#include <queue>
#include <sys/stat.h>
#include <filesystem>

using namespace std;
using namespace OpenMesh;
using namespace Eigen;

string root_path = "/root/original/HW5/";

struct MyTraits : public OpenMesh::DefaultTraits
{
	HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge);
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> Mesh;


Mesh Read_mesh(string name)
{

	IO::Options opt;
	opt += IO::Options::VertexNormal;
	opt += IO::Options::FaceNormal;

	Mesh mesh;
	mesh.request_face_normals();
	mesh.request_vertex_normals();

	cout << "Reading from mesh " << root_path + "data/" + name + ".obj" << "...\n";
	if (!IO::read_mesh(mesh, root_path + "data/" + name + ".obj", opt)) {
		cout << "Read failed.\n";
		exit(0);
	}

	cout << "Mesh stats:\n";
	cout << '\t' << mesh.n_vertices() << " vertices.\n";
	cout << '\t' << mesh.n_edges() << " edges.\n";
	cout << '\t' << mesh.n_faces() << " faces.\n";

	mesh.update_normals();

	return mesh;
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



void visualization(Mesh mesh, const string file_Name, MatrixXd descriptor, int c,string pre_fix) {
	// set color attribute
	mesh.request_face_colors();
	IO::Options opt;
	opt += IO::Options::FaceColor;

	// normalize the color information

	const VectorXd col = descriptor.col(c);
	const double min_value = col.minCoeff();
	const VectorXd shift = col.array() + abs(min_value);

	const double max_value = shift.maxCoeff();
	const VectorXd palette = shift * (255.0 / max_value);

	// Change the color information by iterating the vertice in all faces
	for (Mesh::ConstFaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it)
	{
		Mesh::Color ori_color = { 255, 255, 255 };  // inital color
		for (Mesh::ConstFaceVertexIter fv_it = mesh.cfv_iter(*f_it); fv_it.is_valid(); ++fv_it)
		{
			Mesh::VertexHandle v_i = (*fv_it);
			int idx = v_i.idx();

			ori_color.data()[0] -= palette.coeff(idx) / 3.0;
			ori_color.data()[1] -= palette.coeff(idx) / 3.0;
		}
		mesh.set_color(*f_it, ori_color);
	}
	string output_path = root_path + "Results/"+ pre_fix+"/"+ file_Name +"_eig_" + to_string(c) + ".obj";
	cout << "Writing to file '" << output_path << "'... ";
	if (!IO::write_mesh(mesh, output_path, opt)) {
		cout << "\t can not save the output file!" << endl;
	}
	mesh.release_face_colors();
}


void report_all()
{
	cout << "start to visualize all meshes" << endl;
	Mesh mesh = Read_mesh("target");
	MatrixXd eig_vect = load_csv(root_path + "Results/target_eigvec.csv", 6890, 300);//
	visualization(mesh, "target", eig_vect, 0, "LBO");

	MatrixXd hks_descriptor= load_csv(root_path + "Results/target_hks.csv", 6890, 100);
	visualization(mesh, "target", hks_descriptor, 0, "HKS");
	string source_name;

	for (int i = 1; i < 10; i++)
	{
		source_name = "source_" + to_string(i);
		mesh = Read_mesh(source_name);
		MatrixXd eig_vect = load_csv(root_path + "Results/"+source_name+"_eigvec.csv", 6890, 300);//
		visualization(mesh, source_name, eig_vect, 0, "LBO");
		MatrixXd hks_descriptor = load_csv(root_path + "Results/" + source_name + "_hks.csv", 6890, 100);
		visualization(mesh, source_name, hks_descriptor, 0, "HKS");
	}

}





int main(int argc, char** argv) {
	report_all();
	return 0;
}

