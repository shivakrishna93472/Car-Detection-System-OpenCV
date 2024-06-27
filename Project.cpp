#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Function to detect and return bounding boxes of cars
vector<Rect> detectingtheCars(Mat frame, Ptr<BackgroundSubtractor> pBackSub) {
    Mat fgMask, blurred, thresholded;

    // Apply background subtraction
    pBackSub->apply(frame, fgMask);

    // Blur and threshold the image to reduce noise
    GaussianBlur(fgMask, blurred, Size(15, 15), 0);
    threshold(blurred, thresholded, 200, 255, THRESH_BINARY);

    // Find contours
    vector<vector<Point>> contours;
    findContours(thresholded, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Get bounding boxes
    vector<Rect> boxes;
    for (size_t i = 0; i < contours.size(); i++) {
        Rect box = boundingRect(contours[i]);
        // Filter out small boxes
        if (box.area() > 500) {
            boxes.push_back(box);
        }
    }

    return boxes;
}

int main() {
    // Open video capture
    VideoCapture cap("/home/kpit/Downloads/Cars.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video file" << endl;
        return -1;
    }

    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();

    Mat frame;
    int carCount = 0;
    Point countingLine[2] = {Point(0, 300), Point(800, 300)};
    
    vector<Ptr<Tracker>> trackers;
    vector<Rect> carBoxes;
    vector<int> carIDs;
    int nextID = 0;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;

        vector<Rect> detectedCars = detectingtheCars(frame, pBackSub);

        // Update trackers
        for (size_t i = 0; i < trackers.size();) {
            Rect box;
            if (trackers[i]->update(frame, box)) {
                carBoxes[i] = box;
                i++;
            } else {
                trackers.erase(trackers.begin() + i);
                carBoxes.erase(carBoxes.begin() + i);
                carIDs.erase(carIDs.begin() + i);
            }
        }

        // Add new cars to track
        for (size_t i = 0; i < detectedCars.size(); i++) {
            bool matched = false;
            for (size_t j = 0; j < carBoxes.size(); j++) {
                if ((carBoxes[j] & detectedCars[i]).area() > 0) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                Ptr<Tracker> tracker = TrackerKCF::create();
                tracker->init(frame, detectedCars[i]);
                trackers.push_back(tracker);
                carBoxes.push_back(detectedCars[i]);
                carIDs.push_back(nextID++);
            }
        }

        // Draw bounding boxes and counting line
        for (size_t i = 0; i < carBoxes.size(); i++) {
            rectangle(frame, carBoxes[i], Scalar(0, 255, 0), 2);
            // Check if car crosses the counting line
            if (carBoxes[i].y + carBoxes[i].height / 2 > countingLine[0].y - 2 &&
                carBoxes[i].y + carBoxes[i].height / 2 < countingLine[0].y + 2) {
                // Only count the car if it has not been counted before
                carCount++;
                carIDs[i] = -1;  // Mark this car as counted by setting its ID to -1
            }
        }
        line(frame, countingLine[0], countingLine[1], Scalar(0, 0, 255), 2);

        // Display the result
        putText(frame, "Car Count: " + to_string(carCount), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        imshow("Car Detection", frame);

        if (waitKey(30) >= 0) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
