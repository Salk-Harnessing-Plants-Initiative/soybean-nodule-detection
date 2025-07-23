import argparse
import os
import pandas as pd
import numpy as np


def get_count(plant_df):
    count = len(plant_df)
    return count


def get_height_width(plant_df):
    height = plant_df["y_center"].max() - plant_df["y_center"].min()
    sdy = plant_df["y_center"].std()
    width = plant_df["x_center"].max() - plant_df["x_center"].min()
    sdx = plant_df["x_center"].std()
    sdxy = np.divide(sdx, sdy)
    return height, width, sdy, sdx, sdxy


def get_statistics(plant_trait, prefix):
    if prefix is None:
        prefix = ""

    if len(plant_trait) == 0 or np.isnan(plant_trait).all():
        return {
            f"{prefix}min": np.nan,
            f"{prefix}max": np.nan,
            f"{prefix}mean": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}p5": np.nan,
            f"{prefix}p25": np.nan,
            f"{prefix}p75": np.nan,
            f"{prefix}p95": np.nan,
        }
    else:
        return {
            f"{prefix}min": np.nanmin(plant_trait),
            f"{prefix}max": np.nanmax(plant_trait),
            f"{prefix}mean": np.nanmean(plant_trait),
            f"{prefix}median": np.nanmedian(plant_trait),
            f"{prefix}std": np.nanstd(plant_trait),
            f"{prefix}p5": np.nanpercentile(plant_trait, 5),
            f"{prefix}p25": np.nanpercentile(plant_trait, 25),
            f"{prefix}p75": np.nanpercentile(plant_trait, 75),
            f"{prefix}p95": np.nanpercentile(plant_trait, 95),
        }


def get_traits(yolo_box_csv):
    box_df = pd.read_csv(yolo_box_csv)

    # get the unique plant name and group by each plant to get traits
    plants = box_df["image_path"].unique()
    plants = sorted(plants)
    grouped = box_df.groupby("image_path")

    # # get first 2 groups for test
    # first_2_groups = {}
    # for i, (image_path, group) in enumerate(grouped):
    #     if i < 2:
    #         first_2_groups[image_path] = group
    #     else:
    #         break

    traits_df = pd.DataFrame()
    for (
        image_path,
        plant_df,
    ) in grouped:  # first_2_groups.items() # delete items() for all groups
        print(f"image_path: {image_path}")
        # get box (or nodule) count of each plant
        count = get_count(plant_df)
        # print(f"count: {count}")

        # get height and width of nodule zone
        height, width, sdy, sdx, sdxy = get_height_width(plant_df)
        # print(f"height: {height}, width: {width},sdy: {sdy}, sdx: {sdx}, sdxy: {sdxy}")

        # get summarize of box size
        box_area_plant = plant_df["box_area"]

        prefix = "box_area" + "_"
        box_area_summary = get_statistics(box_area_plant, prefix)
        # print(f"box_area_summary: {box_area_summary}")

        # get summarize of box distribution in verticle axis (y-axis)
        y_center_plant = plant_df["y_center"]

        prefix = "y_center" + "_"
        y_center_summary = get_statistics(y_center_plant, prefix)
        # print(f"y_center_summary: {y_center_summary}")

        data = pd.DataFrame(
            [
                {
                    "image_path": image_path,
                    "count": count,
                    "height": height,
                    "width": width,
                    "sdy": sdy,
                    "sdx": sdx,
                    "sdxy": sdxy,
                    **box_area_summary,
                    **y_center_summary,
                }
            ]
        )
        traits_df = pd.concat([traits_df, data])
    return traits_df


def main():
    parser = argparse.ArgumentParser(description="Get traits from YOLO boxes")
    parser.add_argument(
        "--yolo_box_csv", required=True, help="The yolo detected box prediction"
    )
    parser.add_argument("--traits_csv", required=True, help="csv name of the traits")

    args = parser.parse_args()

    yolo_box_csv = args.yolo_box_csv
    traits_csv = args.traits_csv

    # get traits
    traits_df = get_traits(yolo_box_csv)
    traits_df.to_csv(traits_csv, index=False)


if __name__ == "__main__":
    main()
