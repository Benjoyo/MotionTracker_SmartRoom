package de.bennet_krause.motiontracking;

import java.util.ArrayList;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

/**
 * OpenCV based motion tracker.
 * 
 * @author Bennet
 */
public class MotionTracker implements Runnable {

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    // parameters file for blob detector
    private static final String PARAMS = "/home/pi/NetBeansProjects/OpenCV/dist/params";
    // the blob detector
    private final FeatureDetector blobDetector;
    
    // sensitivity of frame difference
    private final static int SENSITIVITY = 15;
    // pixel size of blur used to smooth the threshold image
    private final static int BLUR_SIZE = 150;

    private final Coordinate coords;

    /**
     * Creates a new motion tracker that updates the x and y coordinates in the given Coordinate object.
     * @param coords 
     */
    public MotionTracker(Coordinate coords) {
        // create a blob detector with the parameters from the param file
        // (especially used to ignore small blobs and those that are very near together)
        blobDetector = FeatureDetector.create(FeatureDetector.SIMPLEBLOB);
        blobDetector.read(PARAMS);
        
        this.coords = coords;
    }

    @Override
    public void run() {

        // the two frames that we compare
        Mat frame1 = new Mat();
        Mat frame2 = new Mat();

        // gray scale of them for creating the difference image
        Mat gray1 = new Mat();
        Mat gray2 = new Mat();

        // "analog" difference image
        Mat differenceImage = new Mat();
        // "digital" difference image
        Mat thresholdImage = new Mat();

        // video capture object with camera as input
        VideoCapture capture;
        capture = new VideoCapture(0);

        if (!capture.isOpened()) {
            System.err.println("Error opening camera!");
        }

        while (true) {
            // read first frame
            capture.read(frame1);

            // exit loop if it's empty, i.e. the camera stream ended
            if (frame1.empty()) {
                System.out.println("Exiting loop!");
                break;
            }

            // gray scale frame1
            Imgproc.cvtColor(frame1, gray1, Imgproc.COLOR_BGR2GRAY);

            // read second frame
            capture.read(frame2);

            // gray scale frame2
            Imgproc.cvtColor(frame2, gray2, Imgproc.COLOR_BGR2GRAY);

            // get the difference between the two captured frames
            Core.absdiff(gray1, gray2, differenceImage);

            // create a binary difference image by thresholding with the given intensity
            Imgproc.threshold(differenceImage, thresholdImage, SENSITIVITY, 255, Imgproc.THRESH_BINARY);

            // blur it to eliminate high frequency camera noise
            Imgproc.blur(thresholdImage, thresholdImage, new Size(BLUR_SIZE, BLUR_SIZE));

            // threshold again because blur output is "analog" again
            Imgproc.threshold(thresholdImage, thresholdImage, SENSITIVITY, 255, Imgproc.THRESH_BINARY);

            // detect big enough blobs in binary difference image and update the position
            getNewPosition(thresholdImage);
        }
    }

    /**
     * Uses FeatureDetector.SIMPLEBLOB to estimate a new position.
     * @param thresholdImage 
     */
    public void getNewPosition(Mat thresholdImage) {

        MatOfKeyPoint matOfKeyPoints = new MatOfKeyPoint();

        blobDetector.detect(thresholdImage, matOfKeyPoints);

        KeyPoint[] keyPoints = matOfKeyPoints.toArray();

        if (keyPoints.length != 0) {
            coords.setX((int) keyPoints[0].pt.x);
            coords.setY((int) keyPoints[0].pt.y);
            // System.out.println("x: " + x + "; y: " + y);
            // Imgproc.rectangle(frame, new Point(x - 25, y - 25), new Point(x + 25, y + 25), new Scalar(0, 255, 0));
            // Imgcodecs.imwrite("frame" + i++ + ".png", frame);
        }
    }

    /**
     * Alternative procedure: less efficient, but also less wobbly.
     *
     * @param thresholdImage
     * @param frame
     */
    public void searchForMotion(Mat thresholdImage, Mat frame) {

        boolean objectDetected;
        Mat temp = new Mat();
        thresholdImage.copyTo(temp);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        Imgproc.findContours(temp, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        
        // if contours vector is not empty, we have found some objects
        objectDetected = contours.size() > 0;

        if (objectDetected) {
            // get the three biggest contours (if any)
            List<MatOfPoint> largestContourVec = new ArrayList<>();
            largestContourVec.add(contours.get(contours.size() - 1));
            if (contours.size() > 1) {
                largestContourVec.add(contours.get(contours.size() - 2));
            }
            if (contours.size() > 2) {
                largestContourVec.add(contours.get(contours.size() - 3));
            }

            int x, y, dist = -1, tempX = 0, tempY = 0;
            // put a rectangle around the largest contours, find their centroids and select
            // the one that is the closest to the last estimated position, this will then be the new one
            for (MatOfPoint mat : largestContourVec) {
                Rect objectBoundingRectangle = Imgproc.boundingRect(mat);
                x = objectBoundingRectangle.x + objectBoundingRectangle.width / 2;
                y = objectBoundingRectangle.y + objectBoundingRectangle.height / 2;

                int newDist = (int) Math.sqrt(Math.pow(x - coords.getX(), 2) + Math.pow(y - coords.getY(), 2));

                if (dist == -1 || newDist < dist) {
                    dist = newDist;
                    tempX = x;
                    tempY = y;
                }
            }

            coords.setX(tempX);
            coords.setY(tempY);

            // System.out.println("x: " + xpos + "; y: " + ypos);
            // Imgproc.rectangle(frame, new Point(xpos - 25, ypos - 25), new Point(xpos + 25, ypos + 25), new Scalar(0, 255, 0));
            // Imgcodecs.imwrite("frame" + i++ + ".png", frame);
        }
    }
}
