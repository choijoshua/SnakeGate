#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>


bool check_color_gate_detection(const cv::Mat& input_img, int x, int y, bool gate) {
    // uchar intensity = input_img.at<uchar>(y, x);
    // if ((int)intensity != 0) {
    //     return true;
    // }

    // return false;

    cv::Vec3b intensity = input_img.at<cv::Vec3b>(y, x);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];

    if (((int)blue + (int)green + (int)red) != 0 ^ !gate) {
        return true;
    }
    return false;
 
}
void calculateHistogram(const cv::Mat& image, std::vector<int>& row_histogram, std::vector<int>& column_histogram) {

    // Initialize histograms
    row_histogram.assign(image.rows, 0);
    column_histogram.assign(image.cols, 0);

    // Iterate through the image to compute histograms
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            // Check if the pixel is white
            if (check_color_gate_detection(image, col, row, true)) {
                row_histogram[row]++;      // Increment the row count
                column_histogram[col]++;  // Increment the column count
                std::cout << "ROW: " << row << " COL: " << col << std::endl;
            }
        }
    }
}

void graphHistogram(const std::vector<int>& hist) {

    for (int i = 0; i < hist.size(); i++) {
        std::cout << "ROW " << i << ": " << hist[i] << std::endl;
    }
} 

int main() {

    cv::Mat image = cv::imread("/home/joshua/Snake_Gate/00636.png", cv::IMREAD_COLOR);
    std::vector<int> row_histogram;
    std::vector<int> col_histogram;

    calculateHistogram(image, row_histogram, col_histogram);
    graphHistogram(row_histogram);


    return 1;

}