#include <OpenMesh/Core/IO/Options.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <cassert>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Tools/Utils/getopt.h>
#include <iostream>
#include<string>
#include<fstream>
#include <limits>
#include <queue>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SymShiftInvert.h>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <iostream>



using namespace std;
using namespace OpenMesh;
using namespace Eigen;
using namespace Spectra;


string root_path = "/root/original/HW5/";

struct MyTraits : public OpenMesh::DefaultTraits
{
	HalfedgeAttributes(OpenMesh::Attributes::PrevHalfedge);
};

typedef OpenMesh::TriMesh_ArrayKernelT<MyTraits> Mesh;

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

Vector3d crossProduct(const Vector3d& A, const Vector3d& B)

{
	Vector3d result;
	result[0] = A[1] * B[2] - A[2] * B[1];
	result[1] = A[2] * B[0] - A[0] * B[2];
	result[2] = A[0] * B[1] - A[1] * B[0];
	return result;
}

Eigen::SparseMatrix<double> Get_cotan(Mesh mesh)
{
	int size = mesh.n_vertices();
	Eigen::SparseMatrix<double> lap_matrix(size, size);
	const float eps = 1e-6f;
	const float cot_max = cos(eps) / sin(eps);

	// get the laplacian matrix
	for (Mesh::VertexIter v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {

		const Mesh::VertexHandle v_i = (*v_it); // handle of the point_i
		const Mesh::Point& p_i = mesh.point(v_i);
		int i = v_i.idx();
		// iterate over all outgoing egdes
		double sum = 0;
		for (Mesh::VertexOHalfedgeIter vf_it = mesh.voh_iter(v_i); vf_it.is_valid(); ++vf_it)// find the neighbohood point J
		{


			const Mesh::HalfedgeHandle edge_ij(*vf_it);// get the edge between the vertice (i,j)
			const Mesh::HalfedgeHandle edge_jk = mesh.next_halfedge_handle(edge_ij); // the edge between (j,k)
			int j = mesh.to_vertex_handle(edge_ij).idx();
			assert(mesh.to_vertex_handle(mesh.next_halfedge_handle(edge_jk)).idx() == i);

			const Mesh::HalfedgeHandle edge_ji = mesh.opposite_halfedge_handle(edge_ij);
			const Mesh::HalfedgeHandle edge_iq = mesh.next_halfedge_handle(edge_ji);

			assert(mesh.to_vertex_handle(mesh.next_halfedge_handle(edge_iq)).idx() == j);

			Mesh::Point& p_j = mesh.point(mesh.to_vertex_handle(edge_ij));
			Mesh::Point& p_k = mesh.point(mesh.to_vertex_handle(edge_jk));
			Mesh::Point& p_q = mesh.point(mesh.to_vertex_handle(edge_iq));

			Vector3d k_to_i(p_i[0] - p_k[0], p_i[1] - p_k[1], p_i[2] - p_k[2]);
			Vector3d k_to_j(p_j[0] - p_k[0], p_j[1] - p_k[1], p_j[2] - p_k[2]);

			Vector3d q_to_i(p_i[0] - p_q[0], p_i[1] - p_q[1], p_i[2] - p_q[2]);
			Vector3d q_to_j(p_j[0] - p_q[0], p_j[1] - p_q[1], p_j[2] - p_q[2]);
			// calculate the positive weight
			float cot1 = abs((k_to_i.dot(k_to_j))) / (1e-6 + crossProduct(k_to_i, k_to_j).norm());
			float cot2 = abs((q_to_i.dot(q_to_j))) / (1e-6 + crossProduct(q_to_i, q_to_j).norm());
			float w_ij = -0.5f * (abs(cot1) + abs(cot2));


			if (isnan(w_ij)) {
				w_ij = 0.0f;
			}


			w_ij = w_ij >= cot_max ? cot_max : w_ij;

			lap_matrix.insert(i, j) = w_ij;
			sum += -w_ij;
		}
		lap_matrix.insert(i, i) = sum;
	}
	return lap_matrix;
}


vector<double>  Get_diag(Mesh mesh)
{
	int size = mesh.n_vertices();
	vector<double> diag(size, size);
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


void Eigen_computing(string obj_name)

{
	IO::Options opt;
	opt += IO::Options::VertexNormal;
	opt += IO::Options::FaceNormal;

	Mesh mesh;
	mesh.request_face_normals();
	mesh.request_vertex_normals();

	cout << "Reading from file " << root_path + "data/" + obj_name + ".obj" << "...\n";
	if (!IO::read_mesh(mesh, root_path + "data/" + obj_name + ".obj", opt)) {
		cout << "Read failed.\n";
		exit(0);
	}

	cout << "Mesh stats:\n";
	cout << '\t' << mesh.n_vertices() << " vertices.\n";
	cout << '\t' << mesh.n_edges() << " edges.\n";
	cout << '\t' << mesh.n_faces() << " faces.\n";

	mesh.update_normals();

	string mesh_name = obj_name;

	mesh_name = mesh_name;


	int nbr_vertex = mesh.n_vertices();
	vector<double> Diag = Get_diag(mesh);
	Eigen::SparseMatrix<double> Cotan_matrix = Get_cotan(mesh);
	//saveData(root_path + "Results/cotan_weight.csv", Cotan_matrix);
	Eigen::SparseMatrix<double> Diag_matrix(nbr_vertex, nbr_vertex);

	for (int i = 0; i < nbr_vertex; i++)
		Diag_matrix.insert(i, i) = Diag[i];

	//Eigen::SparseMatrix<double> lap = Diag_matrix * Cotan_matrix;
	//lap_matrix = Get_laplacian(mesh);
	saveData(root_path + "Results/" + mesh_name + "_diag.csv", Diag_matrix.real());

	//Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> sol(-Diag_matrix*Cotan_matrix);
	//std::cout << sol.eigenvalues().head(20) << std::endl << std::endl;





	//Initialize and compute

	using OpType = SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>;
	using BOpType = SparseSymMatProd<double>;
	OpType op(-Cotan_matrix, Diag_matrix);
	BOpType Bop(Diag_matrix);
	SymGEigsShiftSolver<OpType, BOpType, GEigsMode::ShiftInvert>
		eigs(op, Bop, 301, 650, -1e-5);

	//SparseSymMatProd<double> op(-Cotan_matrix);
	//SparseCholesky<double>  Bop(Diag_matrix);
	//SymGEigsSolver<SparseSymMatProd<double>, SparseCholesky<double>, GEigsMode::Cholesky>
		//eigs(op, Bop, 300, 601);


	eigs.init();
	int nconv = eigs.compute(SortRule::LargestMagn);

	Eigen::VectorXcd evalues;
	Eigen:MatrixXcd evectors;

	if (eigs.info() == CompInfo::Successful)
	{
		evalues = eigs.eigenvalues();
		evectors = eigs.eigenvectors();
	}
	
	
	evectors.real().resize(nbr_vertex, 301);
	ofstream value_file;
	value_file.open(root_path + "Results/" + mesh_name + "_eigval.csv", ios::out);
	MatrixXd save(nbr_vertex, 300);
	//evectors;
	for (int i = 1; i < 301; i++)
	{
		value_file << to_string(evalues(i).real()) << endl;
		save.col(i-1) = evectors.col(i).real();
	}
	
	saveData(root_path + "Results/" + mesh_name + "_eigvec.csv", save.real());
	value_file.close();
}


void Report_all()
{
	Eigen_computing("target");
	string name_root = "source_";
	for (int i = 1; i < 10; i++)
	{
		string name = name_root + to_string(i);
		Eigen_computing(name);

	}

}int main(int argc, char** argv) {
	string str = (argv[1]);

	//Eigen_computing(str);
	Report_all();
	return 0;
}

