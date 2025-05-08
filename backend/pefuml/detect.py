def detect_people_yolov8(image, confidence_threshold=0.3):
    """
    Detect people in the image using YOLOv8.
    Returns bounding boxes of detected people.
    """
    if yolov8_model is None:
        return []

    try:
        # Run detection
        results = yolov8_model(image, conf=confidence_threshold, classes=0)  # class 0 is person in COCO

        # Extract person detections
        person_boxes = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                person_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return person_boxes

    except Exception as e:
        print(f"Error in YOLOv8 person detection: {e}")
        traceback.print_exc()
        return []


def detect_people_mobilenet_ssd(image, confidence_threshold=0.3):
    """
    Detect people in the image using MobileNet SSD.
    Returns bounding boxes of detected people.
    """
    if mobilenet_ssd_model is None:
        return []

    try:
        # Convert image to required format
        input_tensor = tf.convert_to_tensor(image)
        # Add batch dimension
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        detections = mobilenet_ssd_model(input_tensor)

        # Process results
        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(np.int32)
        scores = detections['detection_scores'][0].numpy()

        # Extract person detections (class 1 in COCO)
        person_boxes = []
        h, w = image.shape[0], image.shape[1]

        for i in range(len(scores)):
            if classes[i] == 1 and scores[i] >= confidence_threshold:
                # Convert normalized coordinates to pixel values
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)

                person_boxes.append([x1, y1, x2, y2])

        return person_boxes

    except Exception as e:
        print(f"Error in MobileNet SSD person detection: {e}")
        traceback.print_exc()
        return []


def detect_people_efficientdet(image, confidence_threshold=0.1):
    """
    Detect people in the image using EfficientDet Lite 0.
    Returns bounding boxes of detected people.
    """
    if not person_detector_available or interpreter is None:
        return []

    try:
        # Get input details
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]

        # Get output details - inspect what's actually available
        output_details = interpreter.get_output_details()

        # Print output details for debugging
        print(f"Number of output tensors: {len(output_details)}")
        for i, output in enumerate(output_details):
            print(f"Output {i}: {output['name']} - shape: {output['shape']}")

        # Resize and preprocess the image
        resized_image = cv2.resize(image, (input_width, input_height))

        # Convert to RGB if needed
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and add batch dimension
        input_data = np.expand_dims(resized_image.astype(np.float32) / 255.0, axis=0)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensors - adjusted for EfficientDet Lite 0's actual outputs
        # EfficientDet Lite 0 typically has 4 outputs:
        # - Location boxes (normalized coordinates)
        # - Classes
        # - Scores
        # - Number of detections

        # Safely get outputs
        # The indices might vary based on the model - we'll check each output's name or shape

        # For safety, let's determine which output is which based on shape or name
        boxes_idx, classes_idx, scores_idx, count_idx = None, None, None, None

        # typical shapes from EfficientDet: locations [1,25,4], classes [1,25], scores [1,25], num_detections [1]
        for i, details in enumerate(output_details):
            shape = details['shape']
            if len(shape) == 3 and shape[2] == 4:  # Location boxes [batch, detections, 4 coords]
                boxes_idx = i
            elif len(shape) == 2 and shape[1] > 1:  # Classes or scores [batch, detections]
                if classes_idx is None:
                    classes_idx = i
                elif scores_idx is None:
                    scores_idx = i
            elif len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):  # Number of detections [batch] or [batch, 1]
                count_idx = i

        # If we couldn't identify all outputs, use fixed indices as fallback
        if boxes_idx is None: boxes_idx = 0
        if classes_idx is None: classes_idx = 1
        if scores_idx is None: scores_idx = 2
        if count_idx is None: count_idx = 3

        # Now safely get the outputs
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]

        # Only attempt to get other outputs if the indices are within range
        if classes_idx < len(output_details):
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        else:
            classes = np.ones(len(boxes))  # Default all to class 1

        if scores_idx < len(output_details):
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        else:
            scores = np.ones(len(boxes)) * 0.5  # Default confidence of 0.5

        if count_idx < len(output_details):
            num_detections = int(interpreter.get_tensor(output_details[count_idx]['index'])[0])
        else:
            num_detections = len(boxes)  # Use all boxes

        # Filter for person class (class 0 in COCO)
        person_boxes = []
        for i in range(min(num_detections, len(boxes))):
            if i < len(scores) and scores[i] > confidence_threshold:
                if i < len(classes) and (int(classes[i]) == 0 or int(classes[i]) == 1):  # Allow class 0 or 1 for person
                    # EfficientDet outputs [ymin, xmin, ymax, xmax] normalized
                    h, w = image.shape[0], image.shape[1]

                    # Handle different box formats
                    if len(boxes[i]) == 4:
                        ymin, xmin, ymax, xmax = boxes[i]
                    else:
                        # If unexpected format, use first 4 values and hope they're coordinates
                        ymin, xmin, ymax, xmax = boxes[i][:4]

                    # Convert normalized coordinates to pixel values
                    xmin_px = max(0, int(xmin * w))
                    ymin_px = max(0, int(ymin * h))
                    xmax_px = min(w, int(xmax * w))
                    ymax_px = min(h, int(ymax * h))

                    # Ensure box is valid (non-zero area)
                    if xmin_px < xmax_px and ymin_px < ymax_px:
                        person_boxes.append([xmin_px, ymin_px, xmax_px, ymax_px])

        return person_boxes

    except Exception as e:
        print(f"Error in EfficientDet person detection: {e}")
        traceback.print_exc()
        return []