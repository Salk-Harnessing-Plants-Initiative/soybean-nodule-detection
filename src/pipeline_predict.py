import os
import cv2
import numpy as np
import pandas as pd
import argparse
from ultralytics import YOLO


def adjust_bbox_coordinates(boxes, crop_x, crop_y, edge_threshold, patch_size):
    adjusted_boxes = []
    for box in boxes:
        x_center, y_center, w, h = box.xywh[0]

        # delete box next to the edge (< edge_threshold)
        box_edge = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
        distance_patch = np.array(abs(box_edge - patch_size))
        distance_0 = np.array(abs(box_edge - 0))
        # print(f"box_edge: {box_edge}")
        # print(f"distance: {distance_patch}, edge_threshold: {edge_threshold}")
        if np.all(distance_patch > edge_threshold) and np.all(
            distance_0 > edge_threshold
        ):
            conf = box.conf
            # print(f"conf: {conf}")
            adjusted_boxes.append([x_center + crop_x, y_center + crop_y, w, h, conf])

    return adjusted_boxes


def crop_image(image, crop_size, overlap):
    height, width, _ = image.shape
    print(f"height: {height}, width: {width}")
    crops = []
    for y in range(0, height, crop_size - overlap):
        for x in range(0, width, crop_size - overlap):
            crop = image[y : y + crop_size, x : x + crop_size]
            crops.append((crop, x, y))
    return crops


def predict_image(model, image_name, patch_size, overlap_size, edge_threshold, conf):
    print(f"image_name: {image_name}")
    image = cv2.imread(image_name)
    crops = crop_image(
        image, patch_size, overlap_size
    )  # 512x512 tiles with 64 pixels overlap

    pure_name = os.path.splitext(os.path.basename(image_name))[0]

    # List to store all adjusted bounding boxes
    all_adjusted_boxes = []

    # Get predictions for each crop and adjust the coordinates
    for crop, crop_x, crop_y in crops:
        results = model.predict(crop, conf=conf)

        for result in results:
            adjusted_boxes = adjust_bbox_coordinates(
                result.boxes, crop_x, crop_y, edge_threshold, patch_size
            )
            all_adjusted_boxes.extend(adjusted_boxes)
    return all_adjusted_boxes


def nms(box_df, nms_iou_threshold, conf):
    # Non-Max Suppression
    # box_df columns: image_path, x_center, y_center, w, h, box_area
    # Convert detections to the format expected by NMS: [x1, y1, x2, y2, score]
    # nms_iou_threshold: determines the threshold for deciding whether to suppress
    # a bounding box based on its overlap with another bounding box.
    # If the IoU between two bounding boxes is greater than this threshold,
    # the box with the lower confidence score will be suppressed.
    if len(box_df) > 0:
        box_array = np.array(box_df)
        box_columns = box_df.columns
        boxes = np.array([det[1:5] for det in box_array])  # [1:5]
        scores = np.array([det[-1] for det in box_array])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=conf,
            nms_threshold=nms_iou_threshold,
        )

        final_detections = np.array(
            [box_array[i, :] for i in indices]
        )  # indices.flatten()
        final_detections = pd.DataFrame(final_detections, columns=box_columns)
    else:
        final_detections = pd.DataFrame()
    return final_detections


def add_conf(image, conf, x_center, y_center, w, h):
    text = f"{conf:.2f}"
    text_position = (int(x_center - w / 2) - 10, int(y_center - h / 2) - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)  # Green color
    thickness = 1
    cv2.putText(
        image,
        text,
        text_position,
        font,
        font_scale,
        font_color,
        thickness,
        cv2.LINE_AA,
    )
    return image


def get_boxes(image_name, boxes):
    print(f"len boxes: {len(boxes)}")
    box_df_single = pd.DataFrame()
    for box in boxes:
        x_center, y_center, w, h, conf = map(int, box)
        conf = box[-1].cpu().numpy()[0]
        box_area = w * h
        df = pd.DataFrame(
            {
                "image_path": [image_name],
                "x_center": [x_center],
                "y_center": [y_center],
                "w": [w],
                "h": [h],
                "box_area": [box_area],
                "conf": [conf],
            }
        )

        box_df_single = pd.concat([box_df_single, df])
    return box_df_single


def draw_bboxes(image_name, box_df):
    image = cv2.imread(image_name)
    print(f"image_name: {image_name}")
    print(f"len boxes: {len(box_df)}")

    for i in range(len(box_df)):
        box = box_df.iloc[i]
        x_center, y_center, w, h, conf = (
            box["x_center"],
            box["y_center"],
            box["w"],
            box["h"],
            box["conf"],
        )

        cv2.rectangle(
            image,
            (int(x_center - w / 2), int(y_center - h / 2)),
            (int(x_center + w / 2), int(y_center + h / 2)),
            (0, 0, 255),  # red
            5,
        )

        # add conf above the box
        image = add_conf(image, conf, x_center, y_center, w, h)

    return image, box_df


def draw_bboxes_folder(
    image_folder,
    save_path,
    # box_df,
    patch_size,
    overlap_size,
    model,
    edge_threshold,
    nms_iou_threshold,
    conf,
):
    box_df_folder = pd.DataFrame()
    imageList = [
        os.path.relpath(os.path.join(root, file), image_folder)
        for root, _, files in os.walk(image_folder)
        for file in files
        if (
            file.endswith(".PNG")
            or file.endswith(".png")
            or file.endswith(".jpg")
            or file.endswith(".JPG")
            or file.endswith(".tif")
            or file.endswith(".tiff")
        )
        and not file.startswith(".")
    ]

    for image_name in imageList:
        image = cv2.imread(os.path.join(image_folder, image_name))
        all_adjusted_boxes = predict_image(
            model,
            os.path.join(image_folder, image_name),
            patch_size,
            overlap_size,
            edge_threshold,
            conf,
        )

        # get boxes in each image as df
        box_df_single = get_boxes(image_name, all_adjusted_boxes)
        # apply nms
        box_df_single = nms(box_df_single, nms_iou_threshold, conf)
        # draw boxes of each image
        image_box, box_df_single = draw_bboxes(
            os.path.join(image_folder, image_name), box_df_single
        )

        # save the image with boxes
        save_folder = os.path.join(save_path, "/".join(image_name.split("/")[:-1]))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        new_name = os.path.join(save_path, image_name)
        cv2.imwrite(new_name, image_box)
        box_df_folder = pd.concat([box_df_folder, box_df_single])
    box_df_folder.to_csv(os.path.join(save_path, "box_output.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Training Pipeline")
    parser.add_argument("--img_folder", required=True, help="The original image folder")
    parser.add_argument("--pred_folder", required=True, help="The prediction folder")
    parser.add_argument(
        "--overlap", default=64, help="Overlap pixels between 2 patches"
    )

    args = parser.parse_args()

    img_folder = args.img_folder
    pred_folder = args.pred_folder
    overlap = args.overlap

    # get the well-trained model
    model = YOLO("model/best_70img_4batch_512_64.pt")  # best_30imgs

    # prediction
    # box_df = pd.DataFrame()
    patch_size = 512
    # overlap_size = 0
    edge_threshold = 5
    nms_iou_threshold = (
        0.2  # 0.5 choose a small value because it is less likely to get overlap nodules
    )
    conf = 0.2  # 0.3

    draw_bboxes_folder(
        img_folder,
        pred_folder,
        # box_df,
        patch_size,
        overlap,
        model,
        edge_threshold,
        nms_iou_threshold,
        conf,
    )


if __name__ == "__main__":
    main()
