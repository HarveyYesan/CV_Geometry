#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

int main(int argc, char* argv[])
{
    //reading aligned file
    std::string path_to_dataset = argv[1];
    std::string associate_file = path_to_dataset + "/associate.txt";
    std::ifstream fin(associate_file);
    std::string rgb_file, depth_file, time_rgb, time_depth;
    std::list<cv::Point2f> keypoints; //tracking keypoints
    cv::Mat color;
    cv::Mat depth;
    cv::Mat last_color;

    for(int index = 0; index < 100; index++)
    {
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = cv::imread(path_to_dataset + "/" + rgb_file);
        depth = cv::imread(path_to_dataset + "/" + depth_file, CV_LOAD_IMAGE_UNCHANGED);

        if(index == 0)
        {
            std::vector<cv::KeyPoint> kps;
            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
            detector->detect(color, kps);
            for(auto p:kps)
            {
                keypoints.push_back(p.pt);
            }
            last_color = color;
            continue;            
        }
        if(color.data == nullptr || depth.data == nullptr)
            continue;
        //LK optical flow tracking
        std::vector<cv::Point2f> next_points;
        std::vector<cv::Point2f> prev_points;
        std::vector<unsigned char>status;
        std::vector<float> error;
        for(auto p:keypoints)
        {
            prev_points.push_back(p);
        }
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        cv::calcOpticalFlowPyrLK(last_color, color, prev_points, next_points, status, error);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>( t2-t1 );
        std::cout << "LK use time: " << time_used.count() << " seconds" << std::endl;
        //filter tracking lost points
        int i = 0;
        for(auto iter = keypoints.begin(); iter != keypoints.end(); iter++)
        {
            if(status[i] == 0)
            {
                iter = keypoints.erase(iter);
                continue;
            }
            *iter = next_points[i];
            i++;
        }
        std::cout << "tracked keypoints: " << keypoints.size() << std::endl;
        if(keypoints.size() == 0)
        {
            std::cout << "all keypoints are lost" << std::endl;
        }
        //show
        cv::Mat img_show = color.clone();
        for(auto p:keypoints)
        {
            cv::circle(img_show, p, 10, cv::Scalar(0, 240, 0), 1);
        }
        cv::imshow("lk corners", img_show);
        cv::waitKey(0);
        last_color = color;
    }

    return 0;
}

