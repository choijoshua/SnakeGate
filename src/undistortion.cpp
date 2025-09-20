#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

void undistort_image(cv::Mat& image, cv::Mat& undistorted_image) {
    // undistort cv_image_
    // Camera information
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 285.8460998535156, 0.0, 418.7644958496094, 0.0, 286.0205993652344, 415.0235900878906, 0.0, 0.0, 1.0);
    cv::Mat D = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    // Image size

    // Initialize undistortion maps
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(), K,
                                        image.size(), CV_16SC2, map1,
                                        map2);
    // Undistort in-place by remapping to a temporary buffer and then swapping
    //   cv::Mat undistorted_image;
    cv::remap(image, undistorted_image, map1, map2, cv::INTER_LINEAR);
    // Swap undistorted image content into the original cv_image_->image to avoid
    // extra copying
//   undistorted_image.copyTo(image);
}

int main() {

    cv::Mat image1 = cv::imread("/home/joshua/Snake_Gate/04500.png", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread("/home/joshua/Snake_Gate/04520.png", cv::IMREAD_COLOR);
    cv::Mat image3 = cv::imread("/home/joshua/Snake_Gate/04540.png", cv::IMREAD_COLOR);
    cv::Mat image4 = cv::imread("/home/joshua/Snake_Gate/04560.png", cv::IMREAD_COLOR);
    cv::Mat image5 = cv::imread("/home/joshua/Snake_Gate/04580.png", cv::IMREAD_COLOR);
    cv::Mat image6 = cv::imread("/home/joshua/Snake_Gate/04600.png", cv::IMREAD_COLOR);
    cv::Mat image7 = cv::imread("/home/joshua/Snake_Gate/04620.png", cv::IMREAD_COLOR);
    cv::Mat image8 = cv::imread("/home/joshua/Snake_Gate/04640.png", cv::IMREAD_COLOR);
    cv::Mat image9 = cv::imread("/home/joshua/Snake_Gate/04660.png", cv::IMREAD_COLOR);
    cv::Mat image10 = cv::imread("/home/joshua/Snake_Gate/04680.png", cv::IMREAD_COLOR);

    // New Image
    cv::Mat resized1;
    cv::resize(image1, resized1, cv::Size(848, 800), 0, 0, cv::INTER_LINEAR);
    cv::Mat undistorted_image1;

    undistort_image(resized1, undistorted_image1);
    cv::imshow("Initial Image", undistorted_image1);
    cv::waitKey(0);

    bool success = cv::imwrite("/home/joshua/Snake_Gate/04500.png", undistorted_image1);
    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image." << std::endl;
    }

    // New Image  
    cv::Mat resized2;
    cv::resize(image2, resized2, cv::Size(848, 800), 0, 0, cv::INTER_LINEAR);
    cv::Mat undistorted_image2;

    undistort_image(resized2, undistorted_image2);
    cv::imshow("Initial Image", undistorted_image2);
    cv::waitKey(0);

    success = cv::imwrite("/home/joshua/Snake_Gate/04520.png", undistorted_image2);
    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image." << std::endl;
    }

    // New Image
    cv::Mat resized3;
    cv::resize(image3, resized3, cv::Size(848, 800), 0, 0, cv::INTER_LINEAR);
    cv::Mat undistorted_image3;

    undistort_image(resized3, undistorted_image3);
    cv::imshow("Initial Image", undistorted_image3);
    cv::waitKey(0);

    success = cv::imwrite("/home/joshua/Snake_Gate/04540.png", undistorted_image3);
    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image." << std::endl;
    }

    // New Image
    cv::Mat resized4;
    cv::resize(image4, resized4, cv::Size(848, 800), 0, 0, cv::INTER_LINEAR);
    cv::Mat undistorted_image4;

    undistort_image(resized4, undistorted_image4);
    cv::imshow("Initial Image", undistorted_image4);
    cv::waitKey(0);

    success = cv::imwrite("/home/joshua/Snake_Gate/04560.png", undistorted_image4);
    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image." << std::endl;
    }

    // New Image
    cv::Mat resized5;
    cv::resize(image5, resized5, cv::Size(848, 800), 0, 0, cv::INTER_LINEAR);
    cv::Mat undistorted_image5;

    undistort_image(resized5, undistorted_image5);
    cv::imshow("Initial Image", undistorted_image5);
    cv::waitKey(0);

    success = cv::imwrite("/home/joshua/Snake_Gate/04580.png", undistorted_image5);
    if (success) {
        std::cout << "Image saved successfully!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image." << std::endl;
    }

}