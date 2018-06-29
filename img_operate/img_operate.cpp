#include <iostream>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main ( int argc, char** argv )
{
    // read imgs according to argv[1]
    cv::Mat image;
    image = cv::imread ( argv[1] );
    if ( image.data == nullptr ) //imgs not exits
    {
        cerr<<"file: "<<argv[1]<<"not exits."<<endl;
        return 0;
    }
    
    // read imgs correctly, output img info
    cout<<"img width: "<<image.cols<<",height: "<<image.rows<<",channe: "<<image.channels()<<endl;
    cv::imshow ( "image", image );
    cv::waitKey ( 0 );
    if ( image.type() != CV_8UC1 && image.type() != CV_8UC3 )
    {
        cout<<"please input a gray or color image."<<endl;
        return 0;
    }

    // traverse img, can be used to pixel operating
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for ( size_t y=0; y<image.rows; y++ )
    {
        for ( size_t x=0; x<image.cols; x++ )
        {
            // read pixel at (x,y)
            // use cv::Mat::ptr to get line pointer
            unsigned char* row_ptr = image.ptr<unsigned char> ( y );  // row_ptr is header pointer of line y
            unsigned char* data_ptr = &row_ptr[ x*image.channels() ]; // data_ptr pixel value
            // traverse channel
            for ( int c = 0; c != image.channels(); c++ )
            {
                unsigned char data = data_ptr[c]; // data is value of channel c at I(x,y)
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<"time of traverse image: "<<time_used.count()<<"s."<<endl;

    // copy based on cv::Mat
    // "=" operator can't copy images
    cv::Mat image_another = image;
    // modifying "image_another" could change "image"
    image_another ( cv::Rect ( 0,0,100,100 ) ).setTo ( 0 ); // reset 100*100 pixel at top left corner
    cv::imshow ( "image", image );
    cv::waitKey ( 0 );
    
    // copy img using clone function
    cv::Mat image_clone = image.clone();
    image_clone ( cv::Rect ( 0,0,100,100 ) ).setTo ( 255 );
    cv::imshow ( "image", image );
    cv::imshow ( "image_clone", image_clone );
    cv::waitKey ( 0 );

    cv::destroyAllWindows();
    return 0;
}
