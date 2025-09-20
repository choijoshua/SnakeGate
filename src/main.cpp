#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream> // For saving coordinates to a file

struct Gate {
    std::vector<cv::Point> corners;  // Corner points of the gate
    cv::Point center;               // Center of the gate
    double quality;                 // Quality metric (e.g., area or score)
};

bool is_square(const std::vector<cv::Point>& contour, double& area) {
    // Approximate the contour to a polygon
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);

    // Check if the polygon has 4 sides and is convex
    if (approx.size() == 4 && cv::isContourConvex(approx)) {
        // Check for square-like aspect ratio
        double side1 = cv::norm(approx[0] - approx[1]);
        double side2 = cv::norm(approx[1] - approx[2]);
        double ratio = std::max(side1, side2) / std::min(side1, side2);

        if (ratio < 1.2) {  // Tolerance for being square
            area = cv::contourArea(approx);
            return true;
        }
    }
    return false;
}

void snake_gate_detection(const cv::Mat& input_image, Gate& best_gate, int min_area = 500) {
    best_gate.quality = 0.0;

    // Preprocessing: Convert to grayscale and apply thresholding
    cv::Mat gray, binary;
    cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int k = 0;
    // Loop through contours to find the best gate
    for (const auto& contour : contours) {
        double area;
        if (is_square(contour, area) && area > min_area) {
            k++;
            if (area > best_gate.quality) {  // Choose the largest square-like contour
                best_gate.quality = area;

                // Calculate the center of the gate
                cv::Moments moments = cv::moments(contour);
                best_gate.center = cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);

                // Store the corners
                std::vector<cv::Point> approx;
                cv::approxPolyDP(contour, approx, cv::arcLength(contour, true) * 0.02, true);
                best_gate.corners = approx;
            }
        }
    }
    std::cout << k << std::endl;
}

int main() {
    // Load input image
    cv::Mat input_image = cv::imread("/home/joshua/Snake_Gate00003.png");
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }
    Gate detected_gate;
    // Detect gate
    snake_gate_detection(input_image, detected_gate);

    // Draw and output results
    if (detected_gate.corners.size() == 4) {
        std::ofstream output_file("corner_coordinates.txt");

        // Draw the corner points
        for (const auto& corner : detected_gate.corners) {
            cv::circle(input_image, corner, 5, cv::Scalar(0, 0, 255), -1);  // Red dot for corners
            std::cout << "Corner: (" << corner.x << ", " << corner.y << ")" << std::endl;

            if (output_file.is_open()) {
                output_file << "Corner: (" << corner.x << ", " << corner.y << ")\n";
            }
        }

        // Draw the center point
        cv::circle(input_image, detected_gate.center, 5, cv::Scalar(0, 255, 0), -1);  // Green dot for center
        std::cout << "Center: (" << detected_gate.center.x << ", " << detected_gate.center.y << ")" << std::endl;

        if (output_file.is_open()) {
            output_file << "Center: (" << detected_gate.center.x << ", " << detected_gate.center.y << ")\n";
            output_file.close();
        }

        std::cout << "Gate detected with quality: " << detected_gate.quality << std::endl;
    } else {
        std::cout << "No gate detected." << std::endl;
    }

    // Display the result
    cv::imshow("Gate Detection", input_image);
    cv::waitKey(0);

    return 0;
}
