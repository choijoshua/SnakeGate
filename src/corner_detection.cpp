#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream> // For saving coordinates to a file
#include <time.h>


namespace fs = std::filesystem;

struct Gate {
    std::vector<cv::Point> outer_corners;  // Outer corner points of the gate
    std::vector<cv::Point> inner_corners;  // Inner corner points of the gate
    cv::Point center;               // Center of the gate
    int sz;
    double quality;                 // Quality metric (e.g., area or score)
    int n_sides;
};

void up_and_down(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh);
void left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh);
bool check_color_gate_detection(const cv::Mat& input_img, int x, int y);
bool check_color_not_gate_detection(const cv::Mat& input_img, int x, int y);
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

    bool outer_corners_detected = false;
    srand(time(0));

    for (int i = 0; i < n_samples; i++) {

        x = rand() % im.cols;
        y = rand() % im.rows;
        // std::cout << "X and Y are" << x << ", " << y << std::endl;

        if (check_color_gate_detection(im, x, y)) {
            
            up_and_down(im, x, y, xlow, ylow, xhigh, yhigh);

            if (xlow > xhigh) {
                int temp = xlow;
                xlow = xhigh;
                xhigh = temp;
            }

            sz = yhigh - ylow;
            y = (yhigh + ylow) / 2;

            if (sz > min_pixel) {

                left_and_right(im, xlow, ylow, xlow1, ylow1, xlow2, ylow2);
                left_and_right(im, xhigh, yhigh, xhigh1, yhigh1, xhigh2, yhigh2);

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

                    draw_circle(im, x, y);
                    draw_circle(im, xlow1, ylow1);
                    draw_circle(im, xlow2, ylow2);
                    draw_circle(im, xhigh1, yhigh1);
                    draw_circle(im, xhigh2, yhigh2);

                    outer_corners_detected = true;
                    return;
                }
            }
        }
        
        // if (outer_corners_detected) {
        //     x = best_gate.center.x;
        //     y = best_gate.center.y;



            
        // }
    }
}

void up_and_down(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh) {

    bool done = false;
    ylow = y;
    int x_initial = x;

    // towards negative y
    while (y > 0 && !done) {
        if (check_color_gate_detection(im, x, ylow - 1)) {
            ylow--;
        } else if (ylow - 2 >= 0 && check_color_gate_detection(im, x, ylow - 2)) {
            ylow -= 2;
        } else if (x + 1 < im.cols && check_color_gate_detection(im, x + 1, ylow - 1)) {
            x++;
            ylow--;
        } else if (x - 1 >= 0 && check_color_gate_detection(im, x - 1, ylow - 1)) {
            x--;
            ylow--;
        } else if (x + 2 < im.cols && check_color_gate_detection(im, x + 2, ylow - 1)) {
            x += 2;
            ylow--;
        } else if (x - 2 >= 0 && check_color_gate_detection(im, x - 2, ylow - 1)) {
            x -= 2;
            ylow--;
        } else {
            done = true;
            xlow = x;
        }
    }

    x = x_initial;
    yhigh = y;
    done = false;

    while (yhigh < im.rows - 1 && !done) {
        if (check_color_gate_detection(im, x, yhigh + 1)) {
            yhigh++;
        } else if (yhigh < im.rows - 2 && check_color_gate_detection(im, x, yhigh + 2)) {
            yhigh += 2;
        } else if (x < im.cols - 1 && check_color_gate_detection(im, x + 1, yhigh + 1)) {
            x++;
            yhigh++;
        } else if (x > 0 && check_color_gate_detection(im, x - 1, yhigh + 1)) {
            x--;
            yhigh++;
        } else if (x + 2 < im.cols && check_color_gate_detection(im, x + 2, yhigh + 1)) {
            x += 2;
            yhigh++;
        } else if (x - 1 > 0 && check_color_gate_detection(im, x - 2, yhigh + 1)) {
            x -= 2;
            yhigh++;
        } else {
            done = true;
            xhigh = x;
        }
    }
}

void left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh) {
    bool done = false;
    int y_initial = y;
    xlow = x;

    // snake towards negative x (left)
    while (xlow > 0 && !done) {
        if (check_color_gate_detection(im, xlow - 1, y)) {
            xlow--;
        } else if (xlow > 1 && check_color_gate_detection(im, xlow - 2, y)) {
            xlow -= 2;
        } else if (y < im.rows - 1 && check_color_gate_detection(im, xlow - 1, y + 1)) {
            y++;
            xlow--;
        } else if (y > 0 && check_color_gate_detection(im, xlow - 1, y - 1)) {
            y--;
            xlow--;
        }  else if (y < im.rows - 2 && check_color_gate_detection(im, xlow - 1, y + 2)) {
            y += 2;
            xlow--;
        } else if (y - 1 > 0 && check_color_gate_detection(im, xlow - 1, y - 2)) {
            y -= 2;
            xlow--;
        } else {
            done = true;
            ylow = y;
        }
    }

    y = y_initial;
    xhigh = x;
    done = false;
    // snake towards positive x (right)
    while (xhigh < im.cols - 1 && !done) {
        if (check_color_gate_detection(im, xhigh + 1, y)) {
            xhigh++;
        } else if (xhigh < im.cols - 2 && check_color_gate_detection(im, xhigh + 2, y)) {
            xhigh += 2;
        } else if (y < im.rows- 1 && check_color_gate_detection(im, xhigh + 1, y + 1)) {
            y++;
            xhigh++;
        } else if (y > 0 && check_color_gate_detection(im, xhigh + 1, y - 1)) {
            y--;
            xhigh++;
        }  else if (y < im.rows- 2 && check_color_gate_detection(im, xhigh + 1, y + 2)) {
            y += 2;
            xhigh++;
        } else if (y - 1 > 0 && check_color_gate_detection(im, xhigh + 1, y - 2)) {
            y -= 2;
            xhigh++;
        } else {
            done = 1;
            yhigh = y;
        }
    }
}

void top_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow) {
    return;
}

void top_right(cv::Mat& im, int x, int y, int& xhigh, int& ylow) {
    return;
}

void bottom_left(cv::Mat& im, int x, int y, int& xlow, int& yhigh) {
    return;
}

void bottom_right(cv::Mat& im, int x, int y, int& xhigh, int& yhigh) {
    return;
}

// void check_gate_outline(cv::Mat& input_img, Gate& gate, double quality, int n_sides) {
//     return;
// }

bool check_color_gate_detection(const cv::Mat& input_img, int x, int y) {
    // uchar intensity = input_img.at<uchar>(y, x);
    // if ((int)intensity != 0) {
    //     return true;
    // }

    // return false;

    cv::Vec3b intensity = input_img.at<cv::Vec3b>(y, x);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];
    if (((int)blue + (int)green + (int)red) != 0) {
        return true;
    }
    return false;
}

bool check_color_not_gate_detection(const cv::Mat& input_img, int x, int y) {
    cv::Vec3b intensity = input_img.at<cv::Vec3b>(y, x);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];
    if (((int)blue + (int)green + (int)red) == 0) {
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

void draw_circle(const cv::Mat& input_img, int x, int y) {
    cv::circle(input_img, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
}

void draw_circle(const cv::Mat& input_img, cv::Point point) {
    cv::circle(input_img, point, 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
}

void change_pixel(cv::Mat& input_img, int x, int y) {
    input_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
}

// Function to detect gate corners (placeholder, to be implemented)
void detectGateCorners(const std::string& inputPath, const std::string& outputPath) {
    // Load the input image
    cv::Mat image = cv::imread(inputPath);
    if (image.empty()) {
        std::cerr << "Error: Could not load image at " << inputPath << std::endl;
        return;
    }

    // Process the image (this is where gate detection logic would go)
    Gate gate;
    gate_detection(image, gate, 40, 50);
    // For demonstration, we save the same image to the output
    cv::imwrite(outputPath, image);
}

void processFolder(const std::string& inputFolder, const std::string& outputFolder) {
    // Create the output folder if it doesn't exist
    if (!fs::exists(outputFolder)) {
        fs::create_directories(outputFolder);
    }

    // Iterate through all images in the input folder
    for (const auto& entry : fs::directory_iterator(inputFolder)) {
        const auto& filePath = entry.path();
        if (filePath.extension() == ".png" || filePath.extension() == ".jpg" || filePath.extension() == ".jpeg") {
            std::string inputPath = filePath.string();
            std::string outputPath = (fs::path(outputFolder) / filePath.filename()).string();

            std::cout << "Processing " << inputPath << "..." << std::endl;
            detectGateCorners(inputPath, outputPath);
        }
    }

    std::cout << "Processing complete. Processed images saved in " << outputFolder << "." << std::endl;
}

int main() {

    std::string inputFolder = "/home/joshua/Snake_Gate/masks";
    std::string outputFolder = "/home/joshua/Snake_Gate/snake_gate_detection";
    processFolder(inputFolder, outputFolder);

    return 0;
 
}