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
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

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

void bundleAdjustment(const std::vector<cv::Point3f> points_3d,
                      const std::vector<cv::Point2f> points_2d,
                      const cv::Mat& K, cv::Mat& R, cv::Mat& t)
{
    //init g2o
    //matrix block: variable: 6 dims, error: 3 dims
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    //linear solver: sparse update function
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
    //matrix block solver
    Block* solver_ptr = new Block(linearSolver);
    //optimization method
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    //graphic model
    g2o::SparseOptimizer optimizer;
    //set solver
    optimizer.setAlgorithm(solver);

    //vertex
    //camera pose
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
             R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2),
             R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0,0),
                                                          t.at<double>(1,0),
                                                          t.at<double>(2,0))));
    optimizer.addVertex(pose);

    int index = 1;
    for(const cv::Point3f p:points_3d)
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }    

    g2o::CameraParameters* camera = new g2o::CameraParameters(
        K.at<double>(0,0),
        Eigen::Vector2d(K.at<double>(0,2), K.at<double>(1,2)),
        0);
    camera->setId(0);
    optimizer.addParameter(camera);

    //edges
    index = 1;
    for(const cv::Point2f p:points_2d)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index)));
        edge->setVertex(1, pose);
        edge->setMeasurement(Eigen::Vector2d(p.x, p.y)); //observate value
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity()); //information matrix
        optimizer.addEdge(edge);
        index++;
    }

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    //debugging output
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout << "optimization costs time: " << time_used.count() << " seconds." << std::endl;
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
    cv::Mat d1 = cv::imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point3f> pts_3d;
    std::vector<cv::Point2f> pts_2d;
    for(cv::DMatch m:matches)
    {
        ushort d = d1.ptr<unsigned short> (int (keypoints_1[m.queryIdx].pt.y)) [int (keypoints_1[m.queryIdx].pt.x)];
        if(d == 0)
            continue;
        float dd = d/5000.0;
        cv::Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(cv::Point3f (p1.x*dd, p1.y*dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    std::cout << "3d-2d pairs: " << pts_3d.size() << std::endl;
   
    //rotation vector
    cv::Mat r;
    cv::Mat t;
    cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    //rotation matrix
    cv::Mat R;
    cv::Rodrigues(r, R); 

    std::cout << "R=" << std::endl << R << std::endl;
    std::cout << "t=" << std::endl << t << std::endl;

    //BA
    bundleAdjustment(pts_3d, pts_2d, K, R, t);

    return 0;
}

