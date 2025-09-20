#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <fstream> // For saving coordinates to a file
#include <time.h>

struct Gate {
    std::vector<cv::Point> outer_corners;  // Outer corner points of the gate
    std::vector<cv::Point> inner_corners;  // Inner corner points of the gate
    cv::Point center;               // Center of the gate
    int sz;
    double quality;                 // Quality metric (e.g., area or score)
    int n_sides;
};

void up_and_down(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void top_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void bottom_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void calculateHistogram(const cv::Mat& image, std::vector<int>& row_histogram, std::vector<int>& column_histogram);
bool check_color_gate_detection(const cv::Mat& input_img, int x, int y, bool gate);
void change_pixel(cv::Mat& input_img, int x, int y);
void check_gate_outline(cv::Mat& input_img, Gate& gate, double& quality, int& n_sides);
void draw_circle(const cv::Mat& input_img, cv::Point point);
void draw_circle(const cv::Mat& input_img, int x, int y);


void gate_detection(cv::Mat& im, Gate& best_gate, int n_samples, int min_pixel) {
    int x; 
    int y;
    int xlow;
    int xhigh;
    int ylow;
    int yhigh;
    int xlow1;
    int ylow1;
    int xlow2;
    int ylow2;
    int xhigh1;
    int yhigh1;
    int xhigh2;
    int yhigh2;

    int sz;
    int sz1;
    int sz2;

    bool gate = true;
    srand(time(0));

    for (int i = 0; i < n_samples; i++) {

        x = rand() % im.cols;
        y = rand() % im.rows;
        // std::cout << "X and Y are" << x << ", " << y << std::endl;

        if (check_color_gate_detection(im, x, y, gate)) {
            
            up_and_down(im, x, y, xlow, ylow, xhigh, yhigh, gate);

            if (xlow > xhigh) {
                int temp = xlow;
                xlow = xhigh;
                xhigh = temp;
            }

            sz = yhigh - ylow;
            y = (yhigh + ylow) / 2;

            if (sz > min_pixel) {

                top_left_and_right(im, xlow, ylow, xlow1, ylow1, xlow2, ylow2, gate);
                bottom_left_and_right(im, xhigh, yhigh, xhigh1, yhigh1, xhigh2, yhigh2, gate);

                sz1 = xlow2 - xlow1;
                sz2 = xhigh2 - xhigh1;

                if (sz1 > sz2) {
                    // determine the center x based on the bottom part:
                    x = (xlow2 + xlow1) / 2;
                    // set the size to the largest line found:
                    sz = (sz > sz1) ? sz : sz1;
                } else {
                    // determine the center x based on the top part:
                    x = (xhigh2 + xhigh1) / 2;
                    // set the size to the largest line found:
                    sz = (sz > sz2) ? sz : sz2;
                }

                if (sz1 > min_pixel && sz2 > min_pixel) {
                    
                    // create the gate:
                    best_gate.center = cv::Point(x, y);
                    // store the half gate size:
                    best_gate.sz = sz/2;

                    // The first two corners have a high y:
                    best_gate.outer_corners.emplace_back(cv::Point(xlow1, ylow1));
                    best_gate.outer_corners.emplace_back(cv::Point(xlow2, ylow2));
                    best_gate.outer_corners.emplace_back(cv::Point(xhigh1, yhigh1));
                    best_gate.outer_corners.emplace_back(cv::Point(xhigh2, yhigh2));

                    std::cout << "First corner point is " << xlow1 << ", " << ylow1 << "\n" << std::endl;
                    std::cout << "Second corner point is " << xlow2 << ", " << ylow2 << "\n" << std::endl;
                    std::cout << "Third corner point is " << xhigh1 << ", " << yhigh1 << "\n" << std::endl;
                    std::cout << "Fourth corner point is " << xhigh2 << ", " << yhigh2 << "\n" << std::endl;

                    // draw_circle(im, x, y);
                    draw_circle(im, xlow1, ylow1);
                    draw_circle(im, xlow2, ylow2);
                    draw_circle(im, xhigh1, yhigh1);
                    draw_circle(im, xhigh2, yhigh2);

                    gate = false;   
                }
            
            }
        }
        x = best_gate.center.x;
        y = best_gate.center.y;
        
        if (!gate) {
            if (check_color_gate_detection(im, x, y, gate)) {
                up_and_down(im, x, y, xlow, ylow, xhigh, yhigh, gate);
                
                if (xlow > xhigh) {
                    int temp = xlow;
                    xlow = xhigh;
                    xhigh = temp;
                }

                top_left_and_right(im, xlow, ylow, xlow1, ylow1, xlow2, ylow2, gate);
                bottom_left_and_right(im, xhigh, yhigh, xhigh1, yhigh1, xhigh2, yhigh2, gate);

                best_gate.inner_corners.emplace_back(cv::Point(xlow1, ylow1));
                best_gate.inner_corners.emplace_back(cv::Point(xlow2, ylow2));
                best_gate.inner_corners.emplace_back(cv::Point(xhigh1, yhigh1));
                best_gate.inner_corners.emplace_back(cv::Point(xhigh2, yhigh2));

                std::cout << "First inner corner point is " << xlow1 << ", " << ylow1 << "\n" << std::endl;
                std::cout << "Second inner corner point is " << xlow2 << ", " << ylow2 << "\n" << std::endl;
                std::cout << "Third inner corner point is " << xhigh1 << ", " << yhigh1 << "\n" << std::endl;
                std::cout << "Fourth inner corner point is " << xhigh2 << ", " << yhigh2 << "\n" << std::endl;

                draw_circle(im, x, y);
                draw_circle(im, xlow1, ylow1);
                draw_circle(im, xlow2, ylow2);
                draw_circle(im, xhigh1, yhigh1);
                draw_circle(im, xhigh2, yhigh2);

                return;
            }
        }
    }
}

void up_and_down(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate) {

    bool done = false;
    ylow = y;
    int x_initial = x;

    // towards negative y
    while (y > 0 && !done) {
        if (check_color_gate_detection(im, x, ylow - 1, gate)) {
            ylow--;
        } else if (ylow - 2 >= 0 && check_color_gate_detection(im, x, ylow - 2, gate)) {
            ylow -= 2;
        } else if (x + 1 < im.cols && check_color_gate_detection(im, x + 1, ylow - 1, gate)) {
            x++;
            ylow--;
        } else if (x - 1 >= 0 && check_color_gate_detection(im, x - 1, ylow - 1, gate)) {
            x--;
            ylow--;
        } else if (x + 2 < im.cols && check_color_gate_detection(im, x + 2, ylow - 1, gate)) {
            x += 2;
            ylow--;
        } else if (x - 2 >= 0 && check_color_gate_detection(im, x - 2, ylow - 1, gate)) {
            x -= 2;
            ylow--;
        } else {
            done = true;
            xlow = x;
        }
        change_pixel(im, x, ylow);
    }

    x = x_initial;
    yhigh = y;
    done = false;

    while (yhigh < im.rows - 1 && !done) {
        if (check_color_gate_detection(im, x, yhigh + 1, gate)) {
            yhigh++;
        } else if (yhigh < im.rows - 2 && check_color_gate_detection(im, x, yhigh + 2, gate)) {
            yhigh += 2;
        } else if (x < im.cols - 1 && check_color_gate_detection(im, x + 1, yhigh + 1, gate)) {
            x++;
            yhigh++;
        } else if (x > 0 && check_color_gate_detection(im, x - 1, yhigh + 1, gate)) {
            x--;
            yhigh++;
        } else if (x + 2 < im.cols && check_color_gate_detection(im, x + 2, yhigh + 1, gate)) {
            x += 2;
            yhigh++;
        } else if (x - 1 > 0 && check_color_gate_detection(im, x - 2, yhigh + 1, gate)) {
            x -= 2;
            yhigh++;
        } else {
            done = true;
            xhigh = x;
        }
        change_pixel(im, x, yhigh);
    }
}

void left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate) {
    bool done = false;
    int y_initial = y;
    xlow = x;

    // snake towards negative x (left)
    while (xlow > 0 && !done) {
        if (check_color_gate_detection(im, xlow - 1, y, gate)) {
            xlow--;
        } else if (xlow > 1 && check_color_gate_detection(im, xlow - 2, y, gate)) {
            xlow -= 2;
        } else if (y < im.rows - 1 && check_color_gate_detection(im, xlow - 1, y + 1, gate)) {
            y++;
            xlow--;
        } else if (y > 0 && check_color_gate_detection(im, xlow - 1, y - 1, gate)) {
            y--;
            xlow--;
        }  else if (y < im.rows - 2 && check_color_gate_detection(im, xlow - 1, y + 2, gate)) {
            y += 2;
            xlow--;
        } else if (y - 1 > 0 && check_color_gate_detection(im, xlow - 1, y - 2, gate)) {
            y -= 2;
            xlow--;
        } else {
            done = true;
            ylow = y;
        }
        change_pixel(im, xlow, y);
    }

    y = y_initial;
    xhigh = x;
    done = false;
    // snake towards positive x (right)
    while (xhigh < im.cols - 1 && !done) {
        if (check_color_gate_detection(im, xhigh + 1, y, gate)) {
            xhigh++;
        } else if (xhigh < im.cols - 2 && check_color_gate_detection(im, xhigh + 2, y, gate)) {
            xhigh += 2;
        } else if (y < im.rows- 1 && check_color_gate_detection(im, xhigh + 1, y + 1, gate)) {
            y++;
            xhigh++;
        } else if (y > 0 && check_color_gate_detection(im, xhigh + 1, y - 1, gate)) {
            y--;
            xhigh++;
        }  else if (y < im.rows- 2 && check_color_gate_detection(im, xhigh + 1, y + 2, gate)) {
            y += 2;
            xhigh++;
        } else if (y - 1 > 0 && check_color_gate_detection(im, xhigh + 1, y - 2, gate)) {
            y -= 2;
            xhigh++;
        } else {
            done = 1;
            yhigh = y;
        }

        change_pixel(im, xhigh, y);
    }
}

void top_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate) {
    bool done = false;
    int y_initial = y;
    xlow = x;

    // snake towards negative x (left)
    while (xlow > 0 && !done) {
        if (y > 0 && check_color_gate_detection(im, xlow - 1, y - 1, gate)) {
            y--;
            xlow--;
        } else if (check_color_gate_detection(im, xlow - 1, y, gate)) {
            xlow--;
        } else if (y > 0 && check_color_gate_detection(im, xlow, y - 1, gate)) {
            y--;
        } else if (y < im.rows - 1 && check_color_gate_detection(im, xlow - 1, y + 1, gate)) {
            y++;
            xlow--;
        } else if (y < im.rows - 2 && check_color_gate_detection(im, xlow - 1, y + 2, gate)) {
            y += 2;
            xlow--;
        } else {
            done = true;
            ylow = y;
        }
        change_pixel(im, xlow, y);
    }

    y = y_initial;
    xhigh = x;
    done = false;
    // snake towards positive x (right)
    while (xhigh < im.cols - 1 && !done) {
        if (y > 0 && check_color_gate_detection(im, xhigh + 1, y - 1, gate)) {
            y--;
            xhigh++;
        } else if (check_color_gate_detection(im, xhigh + 1, y, gate)) {
            xhigh++;
        } else if (y - 1 > 0 && check_color_gate_detection(im, xhigh, y - 1, gate)) {
            y--;
        } else if (y < im.rows- 1 && check_color_gate_detection(im, xhigh + 1, y + 1, gate)) {
            y++;
            xhigh++;
        } else if (y < im.rows- 2 && check_color_gate_detection(im, xhigh + 1, y + 2, gate)) {
            y += 2;
            xhigh++;
        } else {
            done = true;
            yhigh = y;
        }
        change_pixel(im, xhigh, y);
    }
}

void bottom_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate) {
    bool done = false;
    int y_initial = y;
    xlow = x;

    // snake towards negative x (left)
    while (xlow > 0 && !done) {
        if (y < im.rows - 1 && check_color_gate_detection(im, xlow - 1, y + 1, gate)) {
            y++;
            xlow--;
        } else if (check_color_gate_detection(im, xlow - 1, y, gate)) {
            xlow--;
        } else if (y < im.rows - 1 && check_color_gate_detection(im, xlow, y + 1, gate)) {
            y++;
        } else if (y > 0 && check_color_gate_detection(im, xlow - 1, y - 1, gate)) {
            y--;
            xlow--;
        } else if (y > 1 && check_color_gate_detection(im, xlow - 1, y - 2, gate)) {
            y-=2;
            xlow--;
        } else {
            done = true;
            ylow = y;
        }
        change_pixel(im, xlow, y);
    }

    y = y_initial;
    xhigh = x;
    done = false;
    // snake towards positive x (right)
    while (xhigh < im.cols - 1 && !done) {
        if (y < im.rows - 1 && check_color_gate_detection(im, xhigh + 1, y + 1, gate)) {
            y++;
            xhigh++;
        } else if (check_color_gate_detection(im, xhigh + 1, y, gate)) {
            xhigh++;
        } else if (y < im.rows - 1 && check_color_gate_detection(im, xhigh, y + 1, gate)) {
            y++;
        } else if (y > 0 && check_color_gate_detection(im, xhigh + 1, y - 1, gate)) {
            y--;
            xhigh++;
        } else if (y + 1 > 0 && check_color_gate_detection(im, xhigh + 1, y - 2, gate)) {
            y -= 2;
            xhigh++;
        } else {
            done = true;
            yhigh = y;
        }
        change_pixel(im, xhigh, y);
    }
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
            }
        }
    }
}


// void check_gate_outline(cv::Mat& input_img, Gate& gate, double quality, int n_sides) {
//     return;
// }

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

void rotate_image(const cv::Mat& input_img, double angle, cv::Mat& rotated_img) {

    // Get the image dimensions
    int width = input_img.cols;
    int height = input_img.rows;

    // Compute the center of rotation
    cv::Point2f center(width / 2.0, height / 2.0);

    // Get the rotation matrix
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // Determine the bounding rectangle after rotation
    // cv::Rect2f boundingBox = cv::RotatedRect(cv::Point2f(0, 0), input_img.size(), angle).boundingRect2f();

    // Adjust the transformation matrix to account for translation
    // rotationMatrix.at<double>(0, 2) += boundingBox.width / 2.0 - center.x;
    // rotationMatrix.at<double>(1, 2) += boundingBox.height / 2.0 - center.y;

    // Rotate the image
    cv::warpAffine(input_img, rotated_img, rotationMatrix, input_img.size());
}

void undistort_image(cv::Mat& image, cv::Mat& undistorted_image) {
    // undistort cv_image_
    // Camera information
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 285.8460998535156, 0.0, 418.7644958496094, 0.0, 286.0205993652344, 415.0235900878906, 0.0, 0.0, 1.0);
    cv::Mat D = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    // Image size

    cv::Size original_size(640, 608);
    cv::Size new_size(848, 800);

    // Scaling factors
    double scale_x = static_cast<double>(new_size.width) / original_size.width;
    double scale_y = static_cast<double>(new_size.height) / original_size.height;

    // Scale the intrinsic matrix
    cv::Mat K_new = K.clone();
    K_new.at<double>(0, 0) *= scale_x; // Scale f_x
    K_new.at<double>(1, 1) *= scale_y; // Scale f_y
    K_new.at<double>(0, 2) *= scale_x; // Scale c_x
    K_new.at<double>(1, 2) *= scale_y; // Scale c_y

    // Initialize undistortion maps
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(K_new, D, cv::Matx33d::eye(), K_new,
                                        new_size, CV_16SC2, map1,
                                        map2);
    // Undistort in-place by remapping to a temporary buffer and then swapping
    //   cv::Mat undistorted_image;
    cv::remap(image, undistorted_image, map1, map2, cv::INTER_LINEAR);
    // Swap undistorted image content into the original cv_image_->image to avoid
    // extra copying
//   undistorted_image.copyTo(image);
}

void draw_circle(const cv::Mat& input_img, int x, int y) {
    cv::circle(input_img, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
}

void draw_circle(const cv::Mat& input_img, cv::Point point) {
    cv::circle(input_img, point, 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
}

void change_pixel(cv::Mat& input_img, int x, int y) {
    input_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
}

int main() {
    // Load input image
    // cv::Mat image = cv::imread("/home/joshua/Snake_Gate/02860.png", cv::IMREAD_COLOR);
    cv::Mat image1 = cv::imread("/home/joshua/Snake_Gate/new_image.png", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread("/home/joshua/Snake_Gate/00206.png", cv::IMREAD_COLOR);
    cv::Mat image3 = cv::imread("/home/joshua/Snake_Gate/00207.png", cv::IMREAD_COLOR);
    cv::Mat image4 = cv::imread("/home/joshua/Snake_Gate/00208.png", cv::IMREAD_COLOR);
    cv::Mat image5 = cv::imread("/home/joshua/Snake_Gate/00209.png", cv::IMREAD_COLOR);
    cv::Mat image6 = cv::imread("/home/joshua/Snake_Gate/00285.png", cv::IMREAD_COLOR);
    cv::Mat image7 = cv::imread("/home/joshua/Snake_Gate/00405.png", cv::IMREAD_COLOR);
    cv::Mat image8 = cv::imread("/home/joshua/Snake_Gate/00489.png", cv::IMREAD_COLOR);
    cv::Mat image9 = cv::imread("/home/joshua/Snake_Gate/00636.png", cv::IMREAD_COLOR);
    // if (image.empty()) {
    //     std::cerr << "Error: Could not load image." << std::endl;
    //     return -1;
    // }

    // cv::Mat rotated_image;

    Gate best_gate1;
    Gate best_gate2;
    Gate best_gate3;
    Gate best_gate4;
    Gate best_gate5;
    Gate best_gate6;
    Gate best_gate7;
    Gate best_gate8;
    Gate best_gate9;

    gate_detection(image1, best_gate1, 400, 100);
    gate_detection(image2, best_gate2, 400, 100);
    gate_detection(image3, best_gate3, 400, 100);
    gate_detection(image4, best_gate4, 400, 100);
    gate_detection(image5, best_gate5, 400, 100);
    gate_detection(image6, best_gate6, 400, 100);
    gate_detection(image7, best_gate7, 400, 100);
    gate_detection(image8, best_gate8, 400, 100);
    gate_detection(image9, best_gate9, 400, 120);

    // cv::Vec3b intensity = image.at<cv::Vec3b>(446, 302);
    // uchar blue = intensity.val[0];
    // uchar green = intensity.val[1];
    // uchar red = intensity.val[2];

    // std::cout << "BLUE: " << (int)blue << " GREEN: " << (int)green << " RED: " << (int)red << std::endl;
    // draw_circle(image, 302, 446);

    // for (int row = 0; row < rotated_image.rows; row++) {

    //     for (int col = 0; col < rotated_image.cols; col++) {

    //         if (check_color_gate_detection(rotated_image, col, row)) {
    //             std::cout << "ROW: " << row << " COL: " << col << std::endl;
    //             rotated_image.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 255, 0);
    //         }

    //     }
    // }

    // Display the result


    cv::imshow("Rotated Image", image1);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image2);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image3);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image4);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image5);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image6);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image7);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image8);
    cv::waitKey(0);
    cv::imshow("Rotated Image", image9);
    cv::waitKey(0);


    return 0;
 
}