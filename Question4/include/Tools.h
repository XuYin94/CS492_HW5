#pragma once
#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Utils/getopt.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include<Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include<iostream>

using namespace Eigen;
using namespace std;
using namespace OpenMesh;

struct MyTraits : public OpenMesh::DefaultTraits
{
	HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge);
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> Mesh;




Vector3d crossProduct(const Vector3d& A, const Vector3d& B)

{
	Vector3d result;
	result[0] = A[1] * B[2] - A[2] * B[1];
	result[1] = A[2] * B[0] - A[0] * B[2];
	result[2] = A[0] * B[1] - A[1] * B[0];
	return result;
}

Eigen::MatrixXd load_csv(std::string path_file, int rows, int cols) {

	std::ifstream reader(path_file);

	std::string line_stream;
	cout << path_file << endl;
	int row = 0;
	int col = 0;
	Eigen::MatrixXd result = Eigen::MatrixXd(rows, cols);
	if (reader.is_open()) {

		while (std::getline(reader, line_stream) && (col < cols)) {

			char* ptr = (char*)line_stream.c_str();
			int len = line_stream.length();
			col = 0;
			char* start = ptr;
			for (int i = 0; i < len; i++) {
				//cout << col << endl;
				if (ptr[i] == ',') {
					//cout << atof(start) << endl;
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




VectorXd Get_diag(string path)
{

	IO::Options opt;
	opt += IO::Options::VertexNormal;
	opt += IO::Options::FaceNormal;

	Mesh mesh;
	mesh.request_face_normals();
	mesh.request_vertex_normals();

	cout << "Reading from file " << path << "...\n";
	if (!IO::read_mesh(mesh, path, opt)) {
		cout << "Read failed.\n";
		exit(0);
	}

	cout << "Mesh stats:\n";
	cout << '\t' << mesh.n_vertices() << " vertices.\n";
	cout << '\t' << mesh.n_edges() << " edges.\n";
	cout << '\t' << mesh.n_faces() << " faces.\n";

	int size = mesh.n_vertices();
	VectorXd diag(size);
	for (Mesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		float area = 0.0;
		Mesh::VertexHandle v_i = (*v_it); // handle of the point_i
		int i = v_i.idx();
		//cout << i << endl;
		for (Mesh::VertexFaceIter vf_it = mesh.vf_iter(v_i); vf_it.is_valid(); ++vf_it)
		{
			Mesh::FaceVertexIter fv_it = mesh.fv_iter(*vf_it);
			const Mesh::Point& P = mesh.point(*fv_it);
			fv_it++;
			const Mesh::Point& Q = mesh.point(*fv_it);
			fv_it++;
			const Mesh::Point& R = mesh.point(*fv_it);
			area += ((Q - P) % (R - P)).norm() * 0.5f / 3;

		}

		diag[i] = area;
	}
	return diag;
}
