#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>

#include <Eigen/Geometry>

int main ()
{
    //3D rotation matrix
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    //rotation vector
    Eigen::AngleAxisd rotation_vector(M_PI/4, Eigen::Vector3d(0,0,1));
    cout.precision(3);
    cout << rotation_vector.matrix() << endl;

    rotation_matrix = rotation_vector.toRotationMatrix();
    cout << rotation_matrix << endl;

    //coordinate transform
    Eigen::Vector3d v(1,0,0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout << v_rotated.transpose() << endl;

    v_rotated = rotation_matrix * v;
    cout << v_rotated.transpose() << endl;

    //euler
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);
    cout << euler_angles << endl;

    //Isometry
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1,3,4));
    cout << "transform matrix" << T.matrix() << endl; 

    Eigen::Vector3d v_transformed = T*v;
    cout << v_transformed.transpose() << endl;

    //Quaternion
    Eigen::Quaterniond q = Eigen::Quaterniond (rotation_vector);
    cout << "Quaternion=" << q.coeffs() << endl;

    q = Eigen::Quaterniond (rotation_matrix);
    cout << "Quaternion=" << q.coeffs() << endl;

    v_rotated = q*v;
    cout << v_rotated.transpose() << endl;

    return 0;

}
