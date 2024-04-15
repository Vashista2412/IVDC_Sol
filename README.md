IVDC (Autonomy) 

I've uploaded files for the 1st and 4th problem statements. I've saved it as Ps1 for 1st and Ps4 for the last.

Computer Vision and ML based PS 

How do different types of filters (e.g., Gaussian, Sobel, Laplacian) impact image processing in computer vision, and what are their specific applications? 

Filtering in image processing is a fundamental technique used to enhance the quality of images. It involves the application of mathematical operations to an image to extract notable features, remove noise, or blur images.1 

Gaussian Blur: 

Commonly used for image smoothing and noise reduction. This filter gives more weight to nearby pixels and less weight to distant pixels, creating a smoother and more natural-looking image. It is commonly used in computer vision, image processing, and signal processing applications.  

Laplacian Filter: 

The Laplacian filter is a linear filter that emphasizes edges in images by taking the second derivative of the picture intensity values. It is widely used for edge detection and may be used with other filters to improve picture attributes.  

Mean Filter: 

The mean filter is a linear filter that smooths an image by replacing each pixel with the average of its neighboring pixels. It is commonly used to remove noise from images but can also blur the image and reduce image detail.2 

Median Filter 

The median filter is a non-linear filter that replaces each pixel with the median value of its local neighborhood. It is commonly used to remove salt and pepper noise (caused by random fluctuations in pixel values.) from images while preserving image detail and sharpness.3 

Sobel Filter: 

Sobel filters are used for edge detection in images. They highlight edges by computing the gradient magnitude of the image. Sobel filters apply convolution with small kernels to compute the gradient approximation in the x and y directions. The magnitude of this gradient represents the edge strength. 

Canny: 

Canny edge detector is a multi-stage algorithm used for edge detection. It applies Gaussian smoothing, calculates gradients, performs non-maximum suppression, and applies edge tracking by hysteresis to detect and track edges with high accuracy and low error rate. 

 

How do adaptive filters adapt to local variations in images, and what advantages do they offer over fixed filters in tasks like image denoising and edge detection? 

Adaptive filters adapted their kernel shape by considering local image contour variations and gave better results than shift-invariant filters that used fixed-shape kernels. Due to their adaptive nature, they were called amoeba filters.4 Adaptive filters are designed to adapt their behavior based on the local characteristics of an image. Unlike fixed filters, which apply the same filtering operation uniformly across the entire image, adaptive filters adjust their parameters or kernel weights dynamically based on the content of the image. This adaptability allows them to better handle local variations in intensity, texture, and noise levels, leading to improved performance in tasks such as image denoising and edge detection.  

Local Neighborhood Processing: Adaptive filters operate on small local neighborhoods within an image rather than the entire image at once. By considering only the nearby pixels, they can adapt to the variations within these neighborhoods. 

Dynamic Parameter Adjustment: The parameters of adaptive filters, such as kernel size or weights, are adjusted based on the characteristics of the local image region being processed. This adjustment can be based on statistical measures like mean, variance, or gradient magnitude within the neighborhood. 

Noise Robustness: Adaptive filters are effective in suppressing noise while preserving image details. By adapting to the local noise characteristics, they can differentiate between noise and actual image features, leading to better denoising performance compared to fixed filters. 

Edge Preservation: In edge detection tasks, adaptive filters can better preserve edges by adjusting their behavior according to the local edge strength and orientation. This helps in accurately detecting and localizing edges while reducing false detections and smoothing artifacts. 

Improved Contrast Enhancement: Adaptive filters can enhance local contrast by adjusting their filtering parameters based on the local intensity variations. This helps in improving the visual quality of images by enhancing details and textures without over-amplifying noise. 

Flexibility: Adaptive filters offer greater flexibility in adapting to different image characteristics and noise levels compared to fixed filters. They can be tuned or designed to suit specific imaging conditions and applications, making them more versatile. 

Real-time Performance: While adaptive filtering may involve more computational complexity compared to fixed filtering, modern hardware and optimized algorithms enable efficient implementation of adaptive filters, allowing them to be used in real-time applications such as video processing and surveillance systems. 

Overall, the adaptive nature of these filters allows them to better adapt to the local variations in images, leading to improved performance in tasks such as denoising, edge detection, contrast enhancement, and other image processing applications. 

Explain the concept of filter kernels in convolutional neural networks (CNNs), and how altering kernel size and weights influence feature extraction and model performance. 

CNN is a type of deep learning model for processing data that has a grid pattern, such as images, which is inspired by the organization of animal visual cortex and designed to automatically and adaptively learn spatial hierarchies of features, from low- to high-level patterns. CNN is a mathematical construct typically composed of three types of layers (or building blocks): convolution, pooling, and fully connected layers.5 In convolutional neural networks (CNNs), filter kernels are small matrices or tensors that are applied to the input data through a process called convolution. These kernels serve as feature detectors, extracting patterns and features from the input data. Each kernel is associated with a set of weights (parameters) learned during training. Altering kernel size affects the receptive field, capturing different spatial information. Changing kernel weights adjusts how kernels respond to features. Larger kernel sizes capture more spatial information but increase complexity, while professionally trained weights optimize feature extraction and improve model performance. 

Discuss the importance of filter design in real-time computer vision applications, considering factors like computational efficiency, accuracy, and adaptability to varying environmental conditions. 

Filter designs involve creating algorithms or mathematical functions that modify or extract specific information from input data, particularly in the context of image processing and computer vision. Filters can include smoothing filters for noise reduction, edge detection filters for highlighting edges, sharpening filters for enhancing details, frequency domain filters for manipulating image frequencies, morphological filters for shape modification, and adaptive filters that adjust parameters based on local characteristics. Effective filter design involves selecting appropriate types, determining optimal parameters, and evaluating performance in terms of accuracy, computational efficiency, and robustness to noise and artifacts. 

Computational Efficiency: 

Real-time computer vision applications often operate under tight computational constraints, especially when deployed on resource-limited devices such as embedded systems or mobile devices. 

Well-designed filters can help reduce computational complexity by efficiently extracting relevant features from the input data while discarding irrelevant information. 

Accuracy: 

The accuracy of computer vision algorithms heavily relies on the effectiveness of the filters used for tasks such as noise reduction, feature extraction, and object detection. 

Filters must be carefully designed to preserve important information while suppressing noise and irrelevant details. This requires a balance between smoothing or blurring to remove noise and edge preservation to maintain important features. 

Adaptive filtering techniques, where filter parameters are adjusted based on the characteristics of the input data, can improve accuracy by dynamically responding to changes in the environment or scene. 

Adaptability to Varying Environmental Conditions: 

Real-world environments are often dynamic and unpredictable, presenting challenges such as changes in lighting conditions, occlusions, and variations in object appearance. 

Filters designed to be adaptive and robust can help computer vision systems maintain performance across different environmental conditions. This may involve incorporating mechanisms for automatic parameter tuning or using machine learning techniques to adaptively adjust filter behavior based on feedback from the environment. 

Multi-scale and multi-modal filtering approaches, which combine information from different scales or modalities, can enhance adaptability by capturing a more comprehensive representation of the scene. 

Robustness to Noise and Artifacts: 

Real-world images often contain various types of noise and artifacts introduced during image acquisition or transmission. 

Filters must be robust to different kinds of noise, such as Gaussian noise, salt-and-pepper noise, or motion blur, while preserving important image details. 

Robust filtering techniques, such as median filtering or morphological operations, can effectively suppress noise without excessively smoothing the image. 

Edge Preservation: 

Edge detection and preservation are essential for tasks like object boundary delineation and feature extraction. 

Filters should preserve sharp edges and boundaries while removing noise and irrelevant texture. 

Edge-aware filtering techniques, such as bilateral filtering or edge-preserving smoothing, can selectively smooth regions while preserving edges, leading to more accurate edge detection and segmentation. 

5. Go through all the filters which are available and get a thorough understanding why and how these filters are used, if possible, make a note of them. Then make a try to design a filter that can achieve the task below? (reducing intensity of horizontal-power lines) 

Grayscale Filter: Converts a color image to grayscale by averaging the RGB channels. 

Gaussian Blur: Blurs an image to reduce noise and detail. 

Median Filter: Replaces each pixel with the median value of its neighboring pixels, useful for removing salt-and-pepper noise. 

Sobel Filter: Detects edges in an image by computing the gradient magnitude. 

Thresholding: Converts an image to binary based on a threshold value, useful for segmenting an image. 

Dilation: Expands the boundaries of objects in an image. 

Erosion: Shrinks the boundaries of objects in an image. 

Histogram Equalization: Enhances the contrast of an image by redistributing pixel intensities. 

Bilateral Filter: Reduces noise while preserving edges in an image. 

https://drive.google.com/file/d/17P508n8JzZuPVw73TadpNjSUKfwDBilq/view?usp=drive_link 

2) PATH PLANNING ALGORITHMS 

         1. Explore different path planning algorithms and classify each algorithm. Also compare them. 

Dijkstra's and A* guarantee optimality but may be computationally expensive. 

RRT is efficient in high-dimensional spaces but may not always find the optimal path. 

Potential field methods are simple and computationally efficient but can get trapped in local minima. 

Genetic algorithms are flexible and can handle complex environments but may require significant computational resources and tuning. 

 

Search-based Planning 
    ├── Breadth-First Searching (BFS) 
    ├── Depth-First Searching (DFS) 
    ├── Best-First Searching 
    ├── Dijkstra's 
    ├── A* 
    ├── Bidirectional A* 
    ├── Anytime Repairing A* 
    ├── Learning Real-time A* (LRTA*) 
    ├── Real-time Adaptive A* (RTAA*) 
    ├── Lifelong Planning A* (LPA*) 
    ├── Dynamic A* (D*) 
    ├── D* Lite 
    └── Anytime D* 
└── Sampling-based Planning 
    ├── RRT 
    ├── RRT-Connect 
    ├── Extended-RRT 
    ├── Dynamic-RRT 
    ├── RRT* 
    ├── Informed RRT* 
    ├── RRT* Smart 
    ├── Anytime RRT* 
    ├── Closed-Loop RRT* 
    ├── Spline-RRT* 
    ├── Fast Marching Trees (FMT*) 
    └── Batch Informed Trees (BIT*) 

 

       2. Which algorithm do you think could be used by an UGV to navigate a track consisting of obstacles such as trees, shrubs, light posts, street signs, barrels, etc. It should be fast and computationally efficient. 

The Rapidly exploring Random Trees (RRT) algorithm would be a suitable choice. RRT is a probabilistic sampling-based algorithm that efficiently explores the search space by randomly sampling and expanding a tree towards unexplored areas. It is particularly effective in high-dimensional spaces with complex obstacle configurations, making it well-suited for environments with diverse obstacles like those described. Additionally, RRT tends to produce feasible paths quickly, which aligns with the requirement for speed in navigation tasks 

 

Estimation and Localisation: 

Explain the basic point cloud data structure. 

A point cloud is a collection of 3D points in space, typically generated by sensors like LiDAR (Light Detection and Ranging) or depth cameras. Each point in the point cloud represents a specific location in 3D space and may contain additional information such as color, intensity, or reflectance. Here's a basic explanation of the point cloud data structure and how we can process raw LiDAR data into a form that can be processed using Python: 

1. Raw LiDAR Data:  Raw LiDAR data consists of measurements obtained by emitting laser beams from a LiDAR sensor and recording the time it takes for the beams to return after hitting objects in the environment. Each measurement includes information such as the 3D coordinates (x, y, z) of the point where the laser beam hits an object, the intensity of the returned signal, and possibly additional information depending on the sensor. 

2. Organizing Data:  The raw LiDAR data is typically organized as a list of points, where each point is represented by its (x, y, z) coordinates and optionally other attributes. For example, if we have a stationary LiDAR sensor in front of a tree, the LiDAR may capture multiple points on the tree's surface, each representing a different location on the tree. 

3. Processing in Python: To process raw LiDAR data in Python, we can use libraries such as NumPy and Pandas for data manipulation and analysis. We can read the raw LiDAR data from a file or directly from the sensor using appropriate drivers or APIs. 

   - Once the data is loaded, we can perform various tasks such as: 

     - Filtering: Remove points that are too close or too far away or filter out points with low intensity. 

     - Segmentation: Separate points belonging to different objects or surfaces in the environment. 

     - Visualization: Plot the point cloud to visualize the captured scene or objects. 

     - Feature Extraction: Identify features such as edges, corners, or planes in the point cloud. 

     - Object Detection: Detect and classify objects (e.g., trees, buildings, vehicles) in the point cloud.  

4. Example: 

   - For example, we can process the raw LiDAR data captured by a stationary LiDAR in front of a tree by reading the data into Python, filtering out points that don't belong to the tree (e.g., points on the ground), and then analyzing the remaining points to estimate properties of the tree such as its size, shape, and density. 

Overall, the basic point cloud data structure consists of a collection of 3D points, and processing raw LiDAR data in Python involves reading, organizing, and analyzing this data to extract useful information about the environment or objects of interest. 

You are a person testing lidar based algorithm on a controlled environment. Mention some of the precautionary measures you would take. [3] Hint : think of when lidar measurement would fail 

When testing LiDAR-based algorithms in a controlled environment, it's crucial to anticipate potential failure scenarios and take precautionary measures to ensure accurate and reliable results. Here are some precautionary measures to consider: 

1. Obstacle Avoidance: Ensure there are no obstructions within the LiDAR sensor's field of view that could interfere with its measurements. Maintain a clear line of sight between the LiDAR sensor and the objects or surfaces it's scanning to prevent occlusions and ensure accurate readings. 

2. Calibration and Alignment: Regularly calibrate and align the LiDAR sensor to ensure accurate measurements. Verify that the LiDAR sensor is properly mounted and oriented according to the intended scanning direction and angle. 

3. Environmental Factors: Control environmental conditions such as lighting, temperature, and humidity to minimize interference with LiDAR measurements. Avoid testing in adverse weather conditions like rain, fog, or snow, which can affect sensor performance and accuracy. 

4. Reflective Surfaces: Be cautious of highly reflective surfaces such as mirrors, glass, or metallic objects, as they can cause signal reflections and distortions in LiDAR data. Minimize the presence of such surfaces within the LiDAR sensor's scanning range to prevent erroneous measurements. 

5. Sensor Range and Resolution: Stay within the operational range and resolution limits of the LiDAR sensor to ensure accurate data capture. Avoid scenarios where objects are located beyond the sensor's maximum range or resolution, as this can lead to incomplete or distorted measurements. 

6. Data Quality Checks: Implement quality checks and validation procedures to identify and discard outlier measurements or erroneous data points. Verify the consistency and reliability of LiDAR measurements by comparing them with ground truth data or alternative sensor readings. 


The rotation of the vehicle frame w.r.t the lidar frame is denoted by the rotation matrix c. given any point p in the lidar frame and considering only rotation transform it into the vehicle frame. (give the mathematical model) 

The rotation of vehicle frame with respect to the LIDAR frame is represented by rotation matrix Cvl. Given any point Pl in lidar frame, expression that correctly transforms the point into vechile frame is Pv = Cvl.Pl 

67 

 Images are missing here! Please do check the document.

Explain Kalman filter by considering the problem of estimating the 1d problem of a vehicle. (Consider approximate assumptions) [We expect you to understand why Kalman Filter is needed.) 

Assumptions: 

We assume linear system dynamics and measurement models for simplicity. 

Process noise Q and measurement noise R are assumed to be Gaussian 

The Kalman filter is an optimal recursive estimator that estimates the state of a dynamic system from a series of noisy measurements. It operates in two main steps: prediction and update. 

Assume a self-driving car locates itself for 1km on a terrain. Having velocity as input and displacement as output. Here target is to reach exactly 1km (as possible as). Below represents car dynamics. [Out put y is car’s position, x is velocity and input u is throttle.] 

 

Wk represents process noise and represents wind and car velocity variation. ~N(0,Q) Gaussian 

Vk is measurement noise. ~N(0,R) Gaussian (R is variance) 

 

This is car model 

So, intial state estimate is xhat k-1 and the predicted state estimate is xhat k. From measurement we get yk. 

Hence, the best estimate of a car’s position is multiplying xhat k and yk results optimal state estimate. 

 

 

 

C is the measurement matrix which maps the state space to measurement space. 

Xhatk-  is priori estimate or prediction.  

Xhatk  is posteri estimate or update. 

Explore the pros and cons of Kalman filter and explain the need of extended Kalman filter. Complete the ekf algorithm in the given ekf_ps folder. You can refer the folder for more hints on the mathematical algorithm. 

Kalman Filter 

Pros: 

Optimal Estimation: The Kalman filter provides the best linear unbiased estimate of a state based on noisy measurements. 

Efficiency: It's computationally efficient, making it suitable for real-time applications like tracking and navigation systems. 

Versatility: The Kalman filter can handle systems with multiple sources of noise and measurements. 

Adaptability: It can be easily adapted to various types of systems by modifying the state and measurement equations. 

Predictive Capability: It not only estimates the current state but also predicts future states based on the system dynamics. 

Cons: 

Linearity Assumption: The standard Kalman filter assumes that the system and measurement equations are linear. This restricts its applicability to linear systems only. 

Sensitivity to Initial Conditions: Incorrect initial estimates or uncertainties can affect the filter's performance significantly. 

Model Mismatch: Any mismatch between the assumed model and the real-world system can degrade the filter's performance. 

Limited to Gaussian Noise: It assumes that the noise is Gaussian, which might not be the case in many real-world scenarios. 

Extended Kalman Filter (EKF) 

The Extended Kalman filter (EKF) addresses some of the limitations of the standard Kalman filter by allowing for nonlinear system and measurement equations. 

Need for Extended Kalman Filter: 

Nonlinearity: Many real-world systems have nonlinear dynamics and measurements. The EKF can handle these nonlinearities by linearizing them around the current estimate. 

Model Flexibility: EKF allows for a broader range of system and measurement models compared to the standard Kalman filter. 

Improved Accuracy: For nonlinear systems, using an EKF can lead to better estimation accuracy compared to trying to linearize the system and using a standard Kalman filter. 

Adaptability to Complex Systems: Systems like robotics, aerospace applications, and biological systems often have nonlinear dynamics. EKF is well-suited for these applications. 

Cons: 

Computational Complexity: EKF can be more computationally intensive than the standard Kalman filter due to the need for Jacobian matrices and matrix multiplications during the linearization process. 

Linearization Errors: The linearization process introduces errors, especially if the system exhibits high nonlinearity or if the current estimate is far from the true state. 

Initialization: Just like the standard Kalman filter, EKF is sensitive to initial conditions. Incorrect initial estimates can lead to divergence or poor performance. 

https://drive.google.com/file/d/1n3KkkB-ZYnVLFi4ZRyeUMVrO8tQ9TAV_/view?usp=drive_link 
