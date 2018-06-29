#include <iostream>
#include <cmath>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry>

int main()
{
    //FIRST
    Eigen::Vector3d P;
    P << 1,2,3;
    cout << P << endl;

    Eigen::Matrix3d S;
    S << 0,1,0,
         1,0,0,
         0,0,-1;
    cout << S << endl;

    Eigen::Matrix3d R;
    R << 1,2,3,
         4,5,6,
         7,8,9;
    cout << R << endl;

    Eigen::Vector3d P1 = R*P;
    cout << P1 << endl;
    
    //SECOND
    Eigen::Vector3d PP = S*P;
    Eigen::Matrix3d RR = S*R*S;
    Eigen::Vector3d PP1 = RR*PP;
    cout << "ans1: " << endl << PP1 << endl;

    cout << "ans2: " << endl << S*P1 << endl;

    Eigen::Vector3d euler_angle_1 = R.eulerAngles(2,1,0);
    Eigen::Vector3d euler_angle_2 = RR.eulerAngles(2,1,0);

    cout << "euler_angle_1" << endl << euler_angle_1 << endl;
    cout << "euler_angle_2" << endl << euler_angle_2 << endl;

    return 0;



}
