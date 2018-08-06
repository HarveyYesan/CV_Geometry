#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

void pose_estimation_2d2d(std::vector<cv::KeyPoint> keypoints_1,
                          std::vector<cv::KeyPoint> keypoints_2,
                          std::vector<cv::DMatch> matches,
                          cv::Mat& R, cv::Mat& t)
{
    //camera intrinsic parameters
    cv::Mat K = (cv::Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    
    //change match point to vector<Point2f>
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    for(int i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //calculate fundamental matrix
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
    std::cout<<"fundamental_matrix is "<<std::endl<< fundamental_matrix<<std::endl;

    //calculate essential matrix
    cv::Point2d principal_point (325.1, 249.7);
    double focal_length = 512;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout<<"essential_matrix is "<<std::endl<< essential_matrix<<std::endl;

    //calculate homography matrix
    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    std::cout<<"homography_matrix is "<<std::endl<<homography_matrix<<std::endl;

    //recover pose from essential matrix
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R is: " << std::endl << R << std::endl;
    std::cout << "t is: " << std::endl << t << std::endl; 
}

int main(int argc, char* argv[])
{
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);

    std::vector<cv::KeyPoint> keypoints_1;
    std::vector<cv::KeyPoint> keypoints_2;
    std::vector<cv::DMatch> matches;

    //orb feature match
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    std::cout << "match point size: " << matches.size() << std::endl;

    //pose estimation
    cv::Mat R;
    cv::Mat t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //verify E = t_x * R * scale
    cv::Mat t_x = (cv::Mat_<double> (3,3) << 
                   0,                  -t.at<double>(2,0), t.at<double>(1,0),
                   t.at<double>(2,0),  0,                  -t.at<double>(0,0),
                   -t.at<double>(1,0), t.at<double>(0,0),  0);
    std::cout << "t^R=" << std::endl << t_x * R << std::endl;

    //verify epipolar
    cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 ); 
    for(cv::DMatch m : matches)
    {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double> (3,1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double> (3,1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "epipolar constraint = " << d << std::endl;
    }
    return 0;

}

