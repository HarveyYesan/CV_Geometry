#include <iostream>
#include<ctime>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 50

int main(int argc, char* argv[])
{
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;

    Eigen::MatrixXd matrix_x;

    matrix_23 << 1,2,3,4,5,6;
    cout << matrix_23 << endl;

    for(int i=0; i<2; i++)
        for(int j=0; j<3; j++)
        {
            cout << matrix_23(i,j) << endl;
        }

    v_3d << 3,2,1;
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    cout << result << endl;

    matrix_33 = Eigen::Matrix3d::Random();
    cout << matrix_33 << endl;

    cout << matrix_33.transpose() << endl;
    cout << matrix_33.sum() << endl;
    cout << matrix_33.trace() << endl;
    cout << 10*matrix_33 << endl;
    cout << matrix_33.inverse() << endl;
    cout << matrix_33.determinant() << endl;

    Eigen::Matrix<double, 3, 3> matrix_33_test;
    matrix_33_test << 1, -3, 3, 3, -5, 3, 6, -6, 4;
    cout << "test: " << matrix_33_test << endl;
    cout << "values_1: " << (matrix_33_test.transpose()*matrix_33_test).eigenvalues() << endl;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver (matrix_33_test);
    cout << "values: "<<eigen_solver.eigenvalues() << endl;

    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);
    
    clock_t time_stt = clock();
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse()*v_Nd;
    cout << "time: " << 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC <<"ms"<<endl;

    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time: "<< 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC <<"ms"<<endl;

    return 0;
}

