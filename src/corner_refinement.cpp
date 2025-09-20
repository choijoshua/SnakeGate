#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <filesystem>
#include <iostream>
#include <vector>
#include <fstream> // For saving coordinates to a file
#include <time.h>


struct Gate {
    std::vector<cv::Point> outer_corners;  // Outer corner points of the gate
    std::vector<cv::Point> inner_corners;  // Inner corner points of the gate
    std::vector<int> row_histogram;
    std::vector<int> col_histogram;
    cv::Point center;               // Center of the gate
    int sz;
    int hsz;
    int vsz;
    double quality;                 // Quality metric (e.g., area or score)
    int n_sides;
};

void up_and_down(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void top_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);
void bottom_left_and_right(cv::Mat& im, int x, int y, int& xlow, int& ylow, int& xhigh, int& yhigh, bool gate);

void top_left(cv::Mat& im, int x, int y, int& xlow, int& ylow, bool gate);
void top_right(cv::Mat& im, int x, int y, int& xhigh, int& ylow, bool gate);
void bottom_left(cv::Mat& im, int x, int y, int& xlow, int& yhigh, bool gate);
void bottom_right(cv::Mat& im, int x, int y, int& xhigh, int& yhigh, bool gate);

void rotateImageAboutPoint(const cv::Mat& inputImage, cv::Mat& outputImage, float angle, const cv::Point2f& pivot);
void gate_refine_corners(cv::Mat& image, Gate& gate, int size);
void refine_single_corner(cv::Mat& image, Gate& gate, cv::Point& corner, int size, float size_factor);
void calculateHistogram(const cv::Mat& image, std::vector<int>& row_histogram, std::vector<int>& column_histogram);
bool check_color_gate_detection(const cv::Mat& input_img, int x, int y, bool gate);
void set_gate_points(Gate& gate);
void change_pixel(cv::Mat& input_img, int x, int y);
void check_gate_outline(cv::Mat& input_img, Gate& gate, double& quality, int& n_sides);
void draw_circle(const cv::Mat& input_img, cv::Point point);
void draw_circle(const cv::Mat& input_img, int x, int y);
void draw_red_circle(const cv::Mat& input_img, int x, int y);
cv::Point computeMedian(const std::vector<cv::Point>& outer_corners);
float computeRollAngle(const cv::Point& median, std::vector<cv::Point> outer_corners);



void gate_detection(cv::Mat& im, int n_samples, int min_pixel) {
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
    int hsz;
    int vsz;
    int sz1;
    int sz2;

    Gate best_gate;

    bool gate = true;
    srand(time(0));

    std::vector<int> row_histogram, col_histogram;
    calculateHistogram(im, row_histogram, col_histogram);

    int maxValue = INT_MIN;
    y = -1;

    for (size_t i = 0; i < row_histogram.size(); ++i) {
        if (row_histogram[i] > maxValue) {
            maxValue = row_histogram[i];
            y = i; // Update index when a new maximum is found
        }
    }

    maxValue = INT_MIN;
    x = -1;

    for (size_t i = 0; i < col_histogram.size(); ++i) {
        if (col_histogram[i] > maxValue) {
            maxValue = col_histogram[i];
            x = i; // Update index when a new maximum is found
        }
    }
    best_gate.row_histogram = row_histogram;
    best_gate.col_histogram = col_histogram;

    for (int i = 0; i < n_samples; i++) {

        if (!check_color_gate_detection(im, x, y, gate)) {
            x = rand() % im.cols;
            y = rand() % im.rows;
        }
     
        up_and_down(im, x, y, xlow, ylow, xhigh, yhigh, gate);

        if (xlow > xhigh) {
            int temp = xlow;
            xlow = xhigh;
            xhigh = temp;
        }

        vsz = yhigh - ylow;
        y = (yhigh + ylow) / 2;

        if (vsz > min_pixel) {

            top_left_and_right(im, xlow, ylow, xlow1, ylow1, xlow2, ylow2, gate);
            bottom_left_and_right(im, xhigh, yhigh, xhigh1, yhigh1, xhigh2, yhigh2, gate);

            sz1 = xlow2 - xlow1;
            sz2 = xhigh2 - xhigh1;

            if (sz1 > sz2) {
                // determine the center x based on the bottom part:
                x = (xlow2 + xlow1) / 2;
                // set the size to the largest line found:
                hsz = sz1;
            } else {
                // determine the center x based on the top part:
                x = (xhigh2 + xhigh1) / 2;
                // set the size to the largest line found:
                hsz = sz2;
            }

            sz = (hsz > vsz) ? vsz : hsz;

            if (hsz > min_pixel) {
                // create the gate:
                best_gate.center = cv::Point(x, y);
                // store the half gate size:
                best_gate.vsz = vsz/2;
                best_gate.hsz = hsz/2;
                best_gate.sz = sz/2;

                // The first two corners have a high y:
                best_gate.outer_corners.emplace_back(cv::Point(xlow1, ylow1));
                best_gate.outer_corners.emplace_back(cv::Point(xlow2, ylow2));
                best_gate.outer_corners.emplace_back(cv::Point(xhigh1, yhigh1));
                best_gate.outer_corners.emplace_back(cv::Point(xhigh2, yhigh2));

                // std::cout << "First corner point is " << xlow1 << ", " << ylow1 << "\n" << std::endl;
                // std::cout << "Second corner point is " << xlow2 << ", " << ylow2 << "\n" << std::endl;
                // std::cout << "Third corner point is " << xhigh1 << ", " << yhigh1 << "\n" << std::endl;
                // std::cout << "Fourth corner point is " << xhigh2 << ", " << yhigh2 << "\n" << std::endl;

                // draw_red_circle(im, x, y);
                // draw_red_circle(im, xlow1, ylow1);
                // draw_red_circle(im, xlow2, ylow2);
                // draw_red_circle(im, xhigh1, yhigh1);
                // draw_red_circle(im, xhigh2, yhigh2);

                set_gate_points(best_gate);

                // draw_red_circle(im, best_gate.outer_corners[0].x, best_gate.outer_corners[0].y);
                // draw_red_circle(im, best_gate.outer_corners[1].x, best_gate.outer_corners[1].y);
                // draw_red_circle(im, best_gate.outer_corners[2].x, best_gate.outer_corners[2].y);
                // draw_red_circle(im, best_gate.outer_corners[3].x, best_gate.outer_corners[3].y);

                gate_refine_corners(im, best_gate, best_gate.sz);

                // draw_red_circle(im, best_gate.outer_corners[0].x, best_gate.outer_corners[0].y);
                // draw_red_circle(im, best_gate.outer_corners[1].x, best_gate.outer_corners[1].y);
                // draw_red_circle(im, best_gate.outer_corners[2].x, best_gate.outer_corners[2].y);
                // draw_red_circle(im, best_gate.outer_corners[3].x, best_gate.outer_corners[3].y);

                top_left(im, best_gate.outer_corners[0].x, best_gate.outer_corners[0].y, best_gate.outer_corners[0].x, best_gate.outer_corners[0].y, gate);
                top_right(im, best_gate.outer_corners[1].x, best_gate.outer_corners[1].y, best_gate.outer_corners[1].x, best_gate.outer_corners[1].y, gate);
                bottom_left(im, best_gate.outer_corners[2].x, best_gate.outer_corners[2].y, best_gate.outer_corners[2].x, best_gate.outer_corners[2].y, gate);
                bottom_right(im, best_gate.outer_corners[3].x, best_gate.outer_corners[3].y, best_gate.outer_corners[3].x, best_gate.outer_corners[3].y, gate);

                draw_circle(im, best_gate.outer_corners[0].x, best_gate.outer_corners[0].y);
                draw_circle(im, best_gate.outer_corners[1].x, best_gate.outer_corners[1].y);
                draw_circle(im, best_gate.outer_corners[2].x, best_gate.outer_corners[2].y);
                draw_circle(im, best_gate.outer_corners[3].x, best_gate.outer_corners[3].y);
                // gate = false;   
                return;
            }
        
        }
        // x = best_gate.center.x;
        // y = best_gate.center.y;
        
        // if (!gate) {
        //     if (check_color_gate_detection(im, x, y, gate)) {
        //         up_and_down(im, x, y, xlow, ylow, xhigh, yhigh, gate);
                
        //         if (xlow > xhigh) {
        //             int temp = xlow;
        //             xlow = xhigh;
        //             xhigh = temp;
        //         }

        //         top_left_and_right(im, xlow, ylow, xlow1, ylow1, xlow2, ylow2, gate);
        //         bottom_left_and_right(im, xhigh, yhigh, xhigh1, yhigh1, xhigh2, yhigh2, gate);

        //         best_gate.inner_corners.emplace_back(cv::Point(xlow1, ylow1));
        //         best_gate.inner_corners.emplace_back(cv::Point(xlow2, ylow2));
        //         best_gate.inner_corners.emplace_back(cv::Point(xhigh1, yhigh1));
        //         best_gate.inner_corners.emplace_back(cv::Point(xhigh2, yhigh2));

        //         std::cout << "First inner corner point is " << xlow1 << ", " << ylow1 << "\n" << std::endl;
        //         std::cout << "Second inner corner point is " << xlow2 << ", " << ylow2 << "\n" << std::endl;
        //         std::cout << "Third inner corner point is " << xhigh1 << ", " << yhigh1 << "\n" << std::endl;
        //         std::cout << "Fourth inner corner point is " << xhigh2 << ", " << yhigh2 << "\n" << std::endl;

        //         draw_circle(im, x, y);
        //         draw_circle(im, xlow1, ylow1);
        //         draw_circle(im, xlow2, ylow2);
        //         draw_circle(im, xhigh1, yhigh1);
        //         draw_circle(im, xhigh2, yhigh2);

        //         return;
        //     }
        // }
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
    }
}

void top_left(cv::Mat& im, int x, int y, int& xlow, int& ylow, bool gate) { 

    bool done = false;
    ylow = y;
    xlow = x;

    // towards negative y
    while (ylow > 0 && !done) {
        if (check_color_gate_detection(im, xlow, ylow - 1, gate)) {
            ylow--;
        } else if (ylow - 2 >= 0 && check_color_gate_detection(im, xlow, ylow - 2, gate)) {
            ylow -= 2;
        } else if (xlow + 1 < im.cols && check_color_gate_detection(im, xlow + 1, ylow - 1, gate)) {
            xlow++;
            ylow--;
        } else if (xlow - 1 >= 0 && check_color_gate_detection(im, xlow - 1, ylow - 1, gate)) {
            xlow--;
            ylow--;
        } else if (xlow + 2 < im.cols && check_color_gate_detection(im, xlow + 2, ylow - 1, gate)) {
            xlow += 2;
            ylow--;
        } else if (xlow - 2 >= 0 && check_color_gate_detection(im, xlow - 2, ylow - 1, gate)) {
            xlow -= 2;
            ylow--;
        } else {
            done = true;
        }
    }
    done = false;

    // snake towards negative x (left)
    while (xlow > 0 && !done) {
        if (ylow > 0 && check_color_gate_detection(im, xlow - 1, ylow - 1, gate)) {
            ylow--;
            xlow--;
        } else if (ylow > 0 && check_color_gate_detection(im, xlow, ylow - 1, gate)) {
            ylow--;
        } else if (check_color_gate_detection(im, xlow - 1, ylow, gate)) {
            xlow--;
        } else if (ylow < im.rows - 1 && check_color_gate_detection(im, xlow - 1, ylow + 1, gate)) {
            ylow++;
            xlow--;
        } else {
            done = true;
        }
    }
}

void top_right(cv::Mat& im, int x, int y, int& xhigh, int& ylow, bool gate) {
    bool done = false;
    xhigh = x;
    ylow = y;

    // towards negative y
    while (ylow > 0 && !done) {
        if (check_color_gate_detection(im, xhigh, ylow - 1, gate)) {
            ylow--;
        } else if (ylow - 2 >= 0 && check_color_gate_detection(im, xhigh, ylow - 2, gate)) {
            ylow -= 2;
        } else if (xhigh + 1 < im.cols && check_color_gate_detection(im, xhigh + 1, ylow - 1, gate)) {
            xhigh++;
            ylow--;
        } else if (xhigh - 1 >= 0 && check_color_gate_detection(im, xhigh - 1, ylow - 1, gate)) {
            xhigh--;
            ylow--;
        } else if (xhigh + 2 < im.cols && check_color_gate_detection(im, xhigh + 2, ylow - 1, gate)) {
            xhigh += 2;
            ylow--;
        } else if (xhigh - 2 >= 0 && check_color_gate_detection(im, xhigh - 2, ylow - 1, gate)) {
            xhigh -= 2;
            ylow--;
        } else {
            done = true;
        }
    }

    done = false;

    // snake towards positive x (right)
    while (xhigh < im.cols - 1 && !done) {
        if (ylow > 0 && check_color_gate_detection(im, xhigh + 1, ylow - 1, gate)) {
            ylow--;
            xhigh++;
        } else if (ylow - 1 > 0 && check_color_gate_detection(im, xhigh, ylow - 1, gate)) {
            ylow--;
        } else if (check_color_gate_detection(im, xhigh + 1, ylow, gate)) {
            xhigh++;
        } else if (y < im.rows- 1 && check_color_gate_detection(im, xhigh + 1, ylow + 1, gate)) {
            ylow++;
            xhigh++;
        } else {
            done = true;
        }
    }
}

void bottom_left(cv::Mat& im, int x, int y, int& xlow, int& yhigh, bool gate) {
    bool done = false;
    xlow = x;
    yhigh = y;

    while (yhigh < im.rows - 1 && !done) {
        if (check_color_gate_detection(im, xlow, yhigh + 1, gate)) {
            yhigh++;
        } else if (yhigh < im.rows - 2 && check_color_gate_detection(im, xlow, yhigh + 2, gate)) {
            yhigh += 2;
        } else if (xlow < im.cols - 1 && check_color_gate_detection(im, xlow + 1, yhigh + 1, gate)) {
            xlow++;
            yhigh++;
        } else if (xlow > 0 && check_color_gate_detection(im, xlow - 1, yhigh + 1, gate)) {
            xlow--;
            yhigh++;
        } else if (xlow + 2 < im.cols && check_color_gate_detection(im, xlow + 2, yhigh + 1, gate)) {
            xlow += 2;
            yhigh++;
        } else if (xlow - 1 > 0 && check_color_gate_detection(im, xlow - 2, yhigh + 1, gate)) {
            xlow -= 2;
            yhigh++;
        } else {
            done = true;
        }
    }

    done = false;

    // snake towards negative x (left)
    while (xlow > 0 && !done) {
        if (yhigh < im.rows - 1 && check_color_gate_detection(im, xlow - 1, yhigh + 1, gate)) {
            yhigh++;
            xlow--;
        } else if (yhigh < im.rows - 1 && check_color_gate_detection(im, xlow, yhigh + 1, gate)) {
            yhigh++;
        } else if (check_color_gate_detection(im, xlow - 1, yhigh, gate)) {
            xlow--;
        } else if (yhigh > 0 && check_color_gate_detection(im, xlow - 1, yhigh - 1, gate)) {
            yhigh--;
            xlow--;
        } else if (yhigh > 1 && check_color_gate_detection(im, xlow - 1, yhigh - 2, gate)) {
            yhigh-=2;
            xlow--;
        } else {
            done = true;
        }
    }
}

void bottom_right(cv::Mat& im, int x, int y, int& xhigh, int& yhigh, bool gate) {
    bool done = false;  
    xhigh = x;
    yhigh = y;

    while (yhigh < im.rows - 1 && !done) {
        if (check_color_gate_detection(im, xhigh, yhigh + 1, gate)) {
            yhigh++;
        } else if (yhigh < im.rows - 2 && check_color_gate_detection(im, xhigh, yhigh + 2, gate)) {
            yhigh += 2;
        } else if (xhigh < im.cols - 1 && check_color_gate_detection(im, xhigh + 1, yhigh + 1, gate)) {
            xhigh++;
            yhigh++;
        } else if (xhigh > 0 && check_color_gate_detection(im, xhigh - 1, yhigh + 1, gate)) {
            xhigh--;
            yhigh++;
        } else if (xhigh + 2 < im.cols && check_color_gate_detection(im, xhigh + 2, yhigh + 1, gate)) {
            xhigh += 2;
            yhigh++;
        } else if (xhigh - 1 > 0 && check_color_gate_detection(im, xhigh - 2, yhigh + 1, gate)) {
            xhigh -= 2;
            yhigh++;
        } else {
            done = true;
        }
    }

    done = false;
    // snake towards positive x (right)
    while (xhigh < im.cols - 1 && !done) {
        if (yhigh < im.rows - 1 && check_color_gate_detection(im, xhigh + 1, yhigh + 1, gate)) {
            yhigh++;
            xhigh++;
        } else if (check_color_gate_detection(im, xhigh + 1, yhigh, gate)) {
            xhigh++;
        } else if (yhigh < im.rows - 1 && check_color_gate_detection(im, xhigh, yhigh + 1, gate)) {
            yhigh++;
        } else if (yhigh > 0 && check_color_gate_detection(im, xhigh + 1, yhigh - 1, gate)) {
            yhigh--;
            xhigh++;
        } else if (yhigh + 1 > 0 && check_color_gate_detection(im, xhigh + 1, yhigh - 2, gate)) {
            yhigh -= 2;
            xhigh++;
        } else {
            done = true;
        }
    }

}

void set_gate_points(Gate& gate) {

    float size_factor = 0.8f;
    gate.outer_corners[0].x = gate.center.x - size_factor * gate.hsz;
    gate.outer_corners[0].y = gate.center.y - size_factor * gate.vsz;

    gate.outer_corners[1].x = gate.center.x + size_factor * gate.hsz;
    gate.outer_corners[1].y = gate.center.y - size_factor * gate.vsz;

    gate.outer_corners[2].x = gate.center.x - size_factor * gate.hsz;
    gate.outer_corners[2].y = gate.center.y + size_factor * gate.vsz;

    gate.outer_corners[3].x = gate.center.x + size_factor * gate.hsz;
    gate.outer_corners[3].y = gate.center.y + size_factor * gate.vsz;
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
                // std::cout << "ROW: " << row << " COL: " << col << std::endl;
            }
        }
    }
}

void gate_refine_corners(cv::Mat& image, Gate& gate, int size) {
    float corner_area = 0.6f; // Can be parameterized if needed
    
    // Refine each corner
    for (size_t i = 0; i < gate.outer_corners.size(); i++) {
        refine_single_corner(image, gate, gate.outer_corners[i], size, corner_area);
    }
}

void refine_single_corner(cv::Mat& image, Gate& gate, cv::Point& corner, int size, float size_factor) {
    // Define the search area around the corner
    int x_l = std::max(0, static_cast<int>(corner.x - size * size_factor));
    int x_r = std::min(image.cols - 1, static_cast<int>(corner.x + size * size_factor));
    int y_l = std::max(0, static_cast<int>(corner.y - size * size_factor));
    int y_h = std::min(image.rows - 1, static_cast<int>(corner.y + size * size_factor));

    // cv::rectangle(image, cv::Point(x_l, y_l), cv::Point(x_r, y_h), cv::Scalar(0, 255, 0), 3); // Green outline, thickness = 3

    int x_size = x_r - x_l + 1;
    int y_size = y_h - y_l + 1;

    // Histograms for x and y directions
    std::vector<int> x_hist(x_size, 0);
    std::vector<int> y_hist(y_size, 0);

    // Iterate through the pixels in the search area
    for (int y = y_l; y <= y_h; y++) {
        for (int x = x_l; x <= x_r; x++) {
            if (check_color_gate_detection(image, x, y, true)) {
                x_hist[x - x_l] = gate.col_histogram[x];
                y_hist[y - y_l] = gate.row_histogram[y];
            }
        }
    }

    // Find the best x and y locations based on histograms
    int best_x_loc = std::distance(x_hist.begin(), std::max_element(x_hist.begin(), x_hist.end())) + x_l;
    int best_y_loc = std::distance(y_hist.begin(), std::max_element(y_hist.begin(), y_hist.end())) + y_l;

    // Update the corner location
    corner.x = best_x_loc;
    corner.y = best_y_loc;
}



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

cv::Point computeMedian(const std::vector<cv::Point>& outer_corners)
{
    float xsum = 0.0f;
    float ysum = 0.0f;
    for (const auto& corner : outer_corners) {
        xsum += corner.x;
        ysum += corner.y;
    }

    float count = static_cast<float>(outer_corners.size());
    // Use std::round if you'd like normal rounding, or static_cast<int>(...) to truncate.
    int median_x = static_cast<int>(std::round(xsum / count));
    int median_y = static_cast<int>(std::round(ysum / count));

    return cv::Point(median_x, median_y);
}

float computeRollAngle(const cv::Point& median, std::vector<cv::Point> outer_corners)
{
    // 1. Sort corners by ascending y-coordinate
    std::sort(outer_corners.begin(), outer_corners.end(),
              [](const cv::Point& a, const cv::Point& b) {
                  return a.y < b.y;
              });

    // Helper to convert radians -> degrees
    auto rad2deg = [](float rad) {
        return rad * 180.0f / static_cast<float>(CV_PI);
    };

    // Function to compute angle in degrees from two integer differences
    auto computeAngleDeg = [&](int dx, int dy) -> float {
        // Avoid division by zero if dx == 0
        if (dx == 0) {
            // If x is 0 but y != 0, angle is 90 deg (vertical line)
            return 90.0f; 
        }
        // Use float division for angle
        float ratio = static_cast<float>(std::abs(dy)) / static_cast<float>(std::abs(dx));
        return rad2deg(std::atan(ratio));
    };

    // ------------------
    // 2. Process the top two corners
    // If the first corner is left by x, store as (x1, y1); otherwise swap
    int x1, y1, x2, y2;
    if (outer_corners[0].x < outer_corners[1].x) {
        x1 = outer_corners[0].x - median.x;
        y1 = outer_corners[0].y - median.y;

        x2 = outer_corners[1].x - median.x;
        y2 = outer_corners[1].y - median.y;
    }
    else {
        x2 = outer_corners[0].x - median.x;
        y2 = outer_corners[0].y - median.y;

        x1 = outer_corners[1].x - median.x;
        y1 = outer_corners[1].y - median.y;
    }

    float theta1 = computeAngleDeg(x1, y1);
    float theta2 = computeAngleDeg(x2, y2);

    // The difference is halved
    return (theta2 - theta1) / 2.0f;
}

void rotateImageAboutPoint(const cv::Mat& inputImage,
                           cv::Mat& outputImage,
                           float angle,
                           const cv::Point2f& pivot)
{
    // Convert angle to radians
    float rad = angle * static_cast<float>(CV_PI) / 180.0f;
    float alpha = std::cos(rad);
    float beta  = std::sin(rad);

    // Prepare the rotation matrix around (0, 0)
    // [ alpha  -beta   tx ]
    // [ beta    alpha   ty ]
    cv::Mat M = (cv::Mat_<float>(2, 3) << alpha, -beta, 0,
                                          beta,   alpha, 0);

    // Enforce that the pivot stays at (px, py).
    // We solve for tx, ty so that M * (px, py) = (px, py).
    // => alpha*px - beta*py + tx = px
    // => beta*px  + alpha*py + ty = py
    // Solve for tx, ty:
    float px = pivot.x;
    float py = pivot.y;

    // tx = px - alpha*px + beta*py
    M.at<float>(0, 2) = px - alpha * px + beta * py;
    // ty = py - beta*px - alpha*py
    M.at<float>(1, 2) = py - beta * px - alpha * py;

    // Perform the affine warp
    // Keep the same output size as the original image
    cv::warpAffine(inputImage, outputImage, M, inputImage.size());
}

void undistort_image(cv::Mat& image) {
    // undistort cv_image_
    // Camera information
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 285.8460998535156, 0.0, 418.7644958496094, 0.0, 286.0205993652344, 415.0235900878906, 0.0, 0.0, 1.0);
    cv::Mat D = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    // Image size

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(848, 800), 0, 0, cv::INTER_LINEAR);

    // Initialize undistortion maps
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(K, D, cv::Matx33d::eye(), K,
                                        resized.size(), CV_16SC2, map1,
                                        map2);
    // Undistort in-place by remapping to a temporary buffer and then swapping
    cv::Mat undistorted_image;

    cv::remap(resized, undistorted_image, map1, map2, cv::INTER_LINEAR);
    // Swap undistorted image content into the original cv_image_->image to avoid
    // extra copying
    undistorted_image.copyTo(image);
}

void draw_circle(const cv::Mat& input_img, int x, int y) {
    cv::circle(input_img, cv::Point(x, y), 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
}

void draw_circle(const cv::Mat& input_img, cv::Point point) {
    cv::circle(input_img, point, 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
}

void draw_red_circle(const cv::Mat& input_img, int x, int y) {
    cv::circle(input_img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);  // Green dot for center
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

    undistort_image(image);
    cv::Mat gray = cv::imread(inputPath, cv::IMREAD_GRAYSCALE);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 2. Canny edge detection
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);

    // 3. Morphological operations (dilate then erode)
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edges, edges, kernel, cv::Point(-1, -1), 1);
    cv::erode(edges, edges, kernel, cv::Point(-1, -1), 1);

    // 4. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 5. Check if we found any contours
    if (contours.empty()) {
        std::cout << "No contours detected in " << inputPath << std::endl;
        return;
    }

    // 6. Find the largest contour by area
    double maxArea = 0.0;
    int largestIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestIdx = static_cast<int>(i);
        }
    }

    if (largestIdx < 0) {
        std::cout << "No valid contours found in " << inputPath << std::endl;
        return;
    }

    // 7. Approximate the largest contour
    std::vector<cv::Point> largestContour = contours[largestIdx];
    double epsilon = 0.02 * cv::arcLength(largestContour, true);
    std::vector<cv::Point> approx;
    cv::approxPolyDP(largestContour, approx, epsilon, true);

    cv::Point center = computeMedian(approx);
    float angle = computeRollAngle(center, approx);
    cv::Mat output_image;
    rotateImageAboutPoint(image, output_image, angle, center);

    // Process the image (this is where gate detection logic would go)
    gate_detection(output_image, 40, 20);
    // For demonstration, we save the same image to the output
    cv::imwrite(outputPath, output_image);
}

void processFolder(const std::string& inputFolder, const std::string& outputFolder) {
    // Create the output folder if it doesn't exist
    if (!std::filesystem::exists(outputFolder)) {
        std::filesystem::create_directories(outputFolder);
    }

    // Iterate through all images in the input folder
    for (const auto& entry : std::filesystem::directory_iterator(inputFolder)) {
        const auto& filePath = entry.path();
        if (filePath.extension() == ".png" || filePath.extension() == ".jpg" || filePath.extension() == ".jpeg") {
            std::string inputPath = filePath.string();
            std::string outputPath = (std::filesystem::path(outputFolder) / filePath.filename()).string();

            std::cout << "Processing " << inputPath << "..." << std::endl;
            detectGateCorners(inputPath, outputPath);
        }
    }

    std::cout << "Processing complete. Processed images saved in " << outputFolder << "." << std::endl;
}

int main() {

    // std::string inputFolder = "/home/joshua/Snake_Gate/dataset_seg_241231/labels/segmented";
    // std::string outputFolder = "/home/joshua/Snake_Gate/dataset_seg_241231/labels/snake_gate";

    std::string inputFolder = "/home/joshua/Snake_Gate/masks";
    std::string outputFolder = "/home/joshua/Snake_Gate/snake_gate_corner_refinement";


    processFolder(inputFolder, outputFolder);

    return 0;
 
}