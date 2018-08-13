#include <iostream>
#include <list>
#include <vector>
#include <chrono>
#include <ctime>
#include <climits>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

//once measure value: 3D point of world coordination and corresponding pixel value
struct Measurement
{
    Measurement(Eigen::Vector3d p, float g) : pos_world(p), grayscale(g) { }
    Eigen::Vector3d pos_world;
    float grayscale;
};

inline Eigen::Vector3d project2Dto3D(int x, int y, int d, float fx, float fy, float cx, float cy, float scale)
{
    float zz = float(d) / scale;
    float xx = zz * (x - cx) / fx;
    float yy = zz * (y - cy) / fy;
    return Eigen::Vector3d (xx, yy, zz);
}

inline Eigen::Vector2d project3Dto2D(float x, float y, float z, float fx, float fy, float cx, float cy)
{
    float u = fx * x / z + cx;
    float v = fy * y / z + cy;
    return Eigen::Vector2d (u, v);
}

//project a 3d point into an image plane, the error is photometric error
//an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect: public g2o::BaseUnaryEdge<1, double, g2o::VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3ProjectDirect() {}
    EdgeSE3ProjectDirect(Eigen::Vector3d point,
                         float fx, float fy,
                         float cx, float cy,
                         cv::Mat* image)
        : x_world_(point),
          fx_(fx),
          fy_(fy),
          cx_(cx),
          cy_(cy),
          image_(image)
    {}

    virtual void computeError()
    {
        const g2o::VertexSE3Expmap* v = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
        Eigen::Vector3d x_local = v->estimate().map(x_world_);
        float x = x_local[0] * fx_ / x_local[2] + cx_;
        float y = x_local[0] * fy_ / x_local[2] + cy_;
        //check x, y is in the image
        if(x-4<0 || (x+4)>image_->cols || (y-4)<0 || (y+4)>image_->rows)
        {
            _error(0,0) = 0.0;
            this->setLevel(1);
        }
        else
        {
            _error(0,0) = getPixelValue(x,y) - _measurement;
        }
    }
    //plus in manifold
    virtual void linearizeOplus()
    {
        if(level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        g2o::VertexSE3Expmap* vtx = static_cast<g2o::VertexSE3Expmap*> (_vertices[0]);
        Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_);

        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0 / xyz_trans[2];
        double invz_2 = invz * invz;

        float u = fx_ * x * invz + cx_;
        float v = fy_ * y * invz + cy_;

        //jacobian from se3 to u,v
        //in g2o, the Lie algebra is (omega, epsilon), where omega is so3 and epsilon is the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai(0,0) = -x*y*invz_2 *fx_;
        jacobian_uv_ksai(0,1) = (1+(x*x*invz_2))*fx_;
        jacobian_uv_ksai(0,2) = -y*invz*fx_;
        jacobian_uv_ksai(0,3) = invz *fx_;
        jacobian_uv_ksai(0,4) = 0;
        jacobian_uv_ksai(0,5) = -x*invz_2 *fx_;

        jacobian_uv_ksai(1,0) = (1+y*y*invz_2)*fy_;
        jacobian_uv_ksai(1,1) = x*y*invz_2 *fy_;
        jacobian_uv_ksai(1,2) = x*invz *fy_;
        jacobian_uv_ksai(1,3) = 0;
        jacobian_uv_ksai(1,4) = invz *fy_;
        jacobian_uv_ksai(1,5) = -y*invz_2 *fy_;
    
        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
        jacobian_pixel_uv(0,0) = (getPixelValue(u+1,v) - getPixelValue(u-1,v)) / 2;
        jacobian_pixel_uv(0,1) = (getPixelValue(u,v+1) - getPixelValue(u,v-1)) / 2;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
    }

    virtual bool read(std::istream& in) {}
    virtual bool write(std::ostream& out) const {}

protected:
    //get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue(float x, float y)
    {
        uchar* data = & image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float (( 1-xx ) * ( 1-yy ) * data[0] +
                      xx* ( 1-yy ) * data[1] +
                      ( 1-xx ) *yy*data[ image_->step ] +
                      xx*yy*data[image_->step+1]
                     );
    }
public:
    //3D point in the world frame
    Eigen::Vector3d x_world_;
    //camera intrinsics
    float cx_ = 0;
    float cy_ = 0;
    float fx_ = 0;
    float fy_ = 0;
    //reference image
    cv::Mat* image_ = nullptr;
};



//sparse direct method
//input measurement value, new gray img, K; output: pose;
//true value: true or false
bool poseEstimationDirect(const std::vector<Measurement>& measurements,
                          cv::Mat* gray,
                          Eigen::Matrix3f& K,
                          Eigen::Isometry3d& Tcw)
{
    //init
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //vertex
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    //edge
    int index = 1;
    for(Measurement m:measurements)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(
            m.pos_world,
            K(0,0), K(1,1), K(0,2), K(1,2),
            gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(index++);
        optimizer.addEdge(edge);
    }
    std::cout << "edges in graph: " << optimizer.edges().size() << std::endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
}

int main(int argc, char* argv[])
{
    // set rand seed
    srand((unsigned int) time(0));
    std::string path_to_dataset = argv[1];
    std::string associate_file = path_to_dataset + "/associate.txt";
    std::ifstream fin(associate_file);
    std::string rgb_file;
    std::string depth_file;
    std::string time_rgb;
    std::string time_depth;
    cv::Mat color;
    cv::Mat depth;
    cv::Mat gray;
    std::vector<Measurement> measurements;
    float fx = 518.0;
    float fy = 519.0;
    float cx = 325.5;
    float cy = 253.5;
    float depth_scale = 1000.0;
    Eigen::Matrix3f K;
    K << fx,   0.f,  cx,
         0.f,  fy,   cy,
         0.f,  0.f,  1.0f;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    cv::Mat prev_color;

    //the first image as a reference
    for(int index = 0; index < 10; index++)
    {
        std::cout << "loop " << index << std::endl;
        fin >> time_rgb >> rgb_file >> time_depth >> depth_file;
        color = cv::imread(path_to_dataset + "/" + rgb_file);
        depth = cv::imread(path_to_dataset + "/" + depth_file, CV_LOAD_IMAGE_UNCHANGED);
        if(color.data==nullptr || depth.data==nullptr)
            continue;
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        if(index == 0)
        {
            //select the pixels with high gradient
            for(int x = 10; x < gray.cols-10; x++)
                for(int y = 10; y < gray.rows-10; y++)
                {
                    Eigen::Vector2d delta(gray.ptr<uchar>(y)[x+1] - gray.ptr<uchar>(y)[x-1],
                                          gray.ptr<uchar>(y+1)[x] - gray.ptr<uchar>(y-1)[x]);
                    if(delta.norm() < 50)
                        continue;
                    ushort d = depth.ptr<ushort>(y)[x];
                    if(d == 0)
                        continue;
                    Eigen::Vector3d p3d = project2Dto3D(x, y, d, fx, fy, cx, cy, depth_scale);
                    float grayscale =float(gray.ptr<uchar>(y)[x]);
                    measurements.push_back(Measurement(p3d, grayscale));
                }
            prev_color = color.clone();
            std::cout << "add total " << measurements.size() << " measurements" << std::endl;
            continue;
        }

        //direct method to estimate pose
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        poseEstimationDirect(measurements, &gray, K, Tcw);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>> (t2 - t1);
        std::cout << "semidense direct method costs time: " << time_used.count() << " second" << std::endl;
        std::cout << "Tcw = " << Tcw.matrix() << std::endl;

        //plot the feature points
        cv::Mat img_show(color.rows*2, color.cols, CV_8UC3);
        prev_color.copyTo(img_show(cv::Rect(0, 0, color.cols, color.rows)));
        color.copyTo(img_show(cv::Rect(0, color.rows, color.cols, color.rows)));
        for(Measurement m:measurements)
        {
            if(rand() > RAND_MAX/5)
                continue;

            Eigen::Vector3d p = m.pos_world;
            Eigen::Vector2d pixel_prev = project3Dto2D(p(0,0), p(1,0), p(2,0), fx, fy, cx, cy);
            Eigen::Vector3d p2 = Tcw * m.pos_world;
            Eigen::Vector2d pixel_now = project3Dto2D(p2(0,0), p2(1,0), p2(2,0), fx, fy, cx, cy);
            if ( pixel_now(0,0)<0 || pixel_now(0,0)>=color.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=color.rows)
                continue;
             
            float b = 255*float(rand()) / RAND_MAX;
            float g = 255*float(rand()) / RAND_MAX;
            float r = 255*float(rand()) / RAND_MAX;
            cv::circle(img_show, cv::Point2d(pixel_prev(0,0), pixel_prev(1,0)), 8, cv::Scalar(b,g,r), 2);
            cv::circle(img_show, cv::Point2d(pixel_now(0,0), pixel_now(1,0)+color.rows), 8, cv::Scalar(b,g,r), 2);
            cv::line(img_show, cv::Point2d(pixel_prev(0,0), pixel_prev(1,0)), cv::Point2d(pixel_now(0,0), pixel_now(1,0)+color.rows), cv::Scalar(b,g,r), 1);

        }
        cv::imshow("result", img_show);
        cv::waitKey(0);

    }
    return 0;
}

