#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

//g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; //eigen align allocater
    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point) : _point(point){}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        _error = _measurement - pose->estimate().map(_point);
    }
    
    virtual void linearizeOplus()
    {
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }
    bool read(std::istream& in) {}
    bool write(std::ostream& out) const {}

protected:
    Eigen::Vector3d _point;
};



void find_feature_matches(const cv::Mat& img_1, const cv::Mat& img_2,
                          std::vector<cv::KeyPoint>& keypoints_1,
                          std::vector<cv::KeyPoint>& keypoints_2,
                          std::vector<cv::DMatch>& matches)
{
    //init
    cv::Mat descriptors_1;
    cv::Mat descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //detect
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //descript
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //match
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    //filter
    double min_dist = 10000.0;
    double max_dist = 0.0;
    for(int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    std::cout << "max dist: " << max_dist << std::endl;
    std::cout << "min dist: " << min_dist << std::endl;

    for(int i = 0; i < descriptors_1.rows; i++)
    {
        if(match[i].distance <= cv::max(2*min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }

}

cv::Point2d pixel2cam(const cv::Point2d& p, const cv::Mat& K)
{
    return cv::Point2d
        (
            (p.x - K.at<double>(0,2)) / K.at<double>(0,0),
            (p.y - K.at<double>(1,2)) / K.at<double>(1,1)
        );
}

void pose_estimation_3d3d(const std::vector<cv::Point3f>& pts1,
                          const std::vector<cv::Point3f>& pts2,
                          cv::Mat& R, cv::Mat& t)
{
    //center of mass
    cv::Point3f p1;
    cv::Point3f p2;
    int N = pts1.size();
    for(int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = cv::Point3f(cv::Vec3f(p1) / N);
    p2 = cv::Point3f(cv::Vec3f(p2) / N);
    //remove the center
    std::vector<cv::Point3f> q1(N);
    std::vector<cv::Point3f> q2(N);
    for(int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }
    //compute W = q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) *
             Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    std::cout << "W=" << W << std::endl;
    //SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd (W, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    std::cout << "U=" << U << std::endl;
    std::cout << "V=" << V << std::endl;
    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) -
                         R_ * Eigen::Vector3d(p1.x, p1.y, p1.z);
    //convert to cv::Mat
    R = (cv::Mat_<double>(3,3) <<
         R_(0,0), R_(0,1), R_(0,2),
         R_(1,0), R_(1,1), R_(1,2),
         R_(2,0), R_(2,1), R_(2,2));
    t = (cv::Mat_<double>(3,1) <<
         t_(0,0), t_(1,0), t_(2,0));
    
}

void bundleAdjustment(const std::vector<cv::Point3f>& pts1,
                      const std::vector<cv::Point3f>& pts2,
                      cv::Mat& R, cv::Mat& t)
{
    //init g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(linearSolver);
    g2o::OptimizationAlgorithmGaussNewton * solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    //vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(Eigen::Matrix3d::Identity(),
                                   Eigen::Vector3d(0,0,0)));
    optimizer.addVertex(pose);

    //edges
    int index = 1;
    std::vector<EdgeProjectXYZRGBDPoseOnly*> edges;
    for(int i = 0; i < pts1.size(); i++)
    {
        EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
            Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(index++);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*> (pose));
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity()*1e4);
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "optimization costs time: " << time_used.count() << "seconds." << std::endl;
    std::cout << "After optimization: " << std::endl;
    std::cout << "T=" << std::endl << Eigen::Isometry3d(pose->estimate()).matrix() << std::endl;
    
}


int main(int argc, char* argv[])
{
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "find " << matches.size() << "orb matched points" << std::endl;

    //generate 3D points
    cv::Mat depth1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat depth2 = cv::imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts1_3d;
    std::vector<cv::Point3f> pts2_3d;
    for(cv::DMatch m:matches)
    {
        ushort d1 = depth1.ptr<unsigned short> (int (keypoints_1[m.queryIdx].pt.y)) [int (keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short> (int (keypoints_2[m.queryIdx].pt.y)) [int (keypoints_2[m.queryIdx].pt.x)];
        if((d1 == 0) || (d2 == 0))
            continue;
        float dd1 = d1/5000.0;
        float dd2 = d2/5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Point2d p2 = pixel2cam(keypoints_2[m.queryIdx].pt, K);
        pts1_3d.push_back(cv::Point3f (p1.x*dd1, p1.y*dd1, dd1));
        pts2_3d.push_back(cv::Point3f (p2.x*dd2, p2.y*dd2, dd2));
    }
    std::cout << "3d-2d pairs: " << pts1_3d.size() << std::endl;

    cv::Mat R;
    cv::Mat t;
    pose_estimation_3d3d(pts1_3d, pts2_3d, R, t);
    std::cout << "ICP via SVD results: " << std::endl;
    std::cout << "R = " << R << std::endl;
    std::cout << "t = " << t << std::endl;
    std::cout << "R_inv = " << R.t() << std::endl;
    std::cout << "t_inv = " << -R.t() * t << std::endl;

    std::cout << "After BA:" << std::endl;
    bundleAdjustment(pts1_3d, pts2_3d, R, t);
   
    //verify p1 = R * p2 + t
    for(int i = 0; i < 5; i++)
    {
        std::cout << "p1 = " << pts1_3d[i] << std::endl;
        std::cout << "p2 = " << pts2_3d[i] << std::endl;
        std::cout << "(R * P2 + t) = " <<
                  R * (cv::Mat_<double>(3,1) << pts2_3d[i].x, pts2_3d[i].y, pts2_3d[i].z) + t
                  << std::endl;

    } 

    return 0;
}

